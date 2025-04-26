package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strings"

	"github.com/ollama/ollama/api"
)

var pythonFuncRegex = regexp.MustCompile(`(\w+)\((.*?)\)`)

func parseObjects(s string) []map[string]any {
	var objs []map[string]any
	for offset := 0; offset < len(s); {
		var obj map[string]any
		decoder := json.NewDecoder(strings.NewReader(s[offset:]))
		err := decoder.Decode(&obj)
		switch {
		case errors.Is(err, io.EOF), errors.Is(err, io.ErrUnexpectedEOF):
			return objs
		case err != nil:
			var syntax *json.SyntaxError
			var unmarshalType *json.UnmarshalTypeError
			switch {
			case errors.As(err, &syntax):
				offset += int(syntax.Offset)
				continue
			case errors.As(err, &unmarshalType):
				offset += int(unmarshalType.Offset)
				continue
			default:
				return nil
			}
		}
		offset += int(decoder.InputOffset())
		objs = append(objs, obj)
	}
	return objs
}

// parsePythonFunctionCall parses Python function calls from a string
// it supports both positional and keyword arguments, as well as multiple functions in a single string
func parsePythonFunctionCall(s string) ([]api.ToolCall, bool) {
	matches := pythonFuncRegex.FindAllStringSubmatchIndex(s, -1)
	if len(matches) == 0 {
		return nil, false
	}

	var toolCalls []api.ToolCall
	for _, match := range matches {
		name := s[match[2]:match[3]]
		args := s[match[4]:match[5]]

		arguments := make(api.ToolCallFunctionArguments)
		if strings.Contains(args, "=") { // Keyword args
			pairs := strings.SplitSeq(args, ",")
			for pair := range pairs {
				pair = strings.TrimSpace(pair)
				kv := strings.Split(pair, "=")
				if len(kv) == 2 {
					key := strings.TrimSpace(kv[0])
					value := strings.TrimSpace(kv[1])
					arguments[key] = value
				}
			}
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      name,
					Arguments: arguments,
				},
			})
		}
	}

	if len(toolCalls) > 0 {
		return toolCalls, true
	}
	return nil, false
}

// TODO: revisit to see if necessary - most do come in this
// ToolCallFormat represents different possible formats for tool calls
type toolCallFormat struct {
	// Direct format
	Name      string         `json:"name,omitempty"`
	Arguments map[string]any `json:"arguments,omitempty"`

	// Command-r-plus format
	ToolName   string         `json:"tool_name,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`

	// Function format
	Function *struct {
		Name       string         `json:"name"`
		Arguments  map[string]any `json:"arguments,omitempty"`
		Parameters map[string]any `json:"parameters,omitempty"`
	} `json:"function,omitempty"`

	// Xlam format
	ToolCalls []toolCallFormat `json:"tool_calls,omitempty"`
}

func parseJSONToolCalls(obj map[string]any) ([]api.ToolCall, bool) {
	// Helper to convert any to []any safely
	toArray := func(v any) []any {
		if arr, ok := v.([]any); ok {
			return arr
		}
		return nil
	}

	// Convert a single format to a tool call
	makeToolCall := func(f toolCallFormat) (api.ToolCall, bool) {
		switch {
		case f.Name != "" && f.Arguments != nil:
			return api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      f.Name,
					Arguments: f.Arguments,
				},
			}, true
		case f.Name != "" && f.Parameters != nil: // Handle parameters field
			return api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      f.Name,
					Arguments: f.Parameters,
				},
			}, true
		case f.ToolName != "" && f.Parameters != nil:
			return api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      f.ToolName,
					Arguments: f.Parameters,
				},
			}, true
		case f.Function != nil && f.Function.Name != "":
			args := f.Function.Arguments
			if args == nil {
				args = f.Function.Parameters
			}
			if args != nil {
				return api.ToolCall{
					Function: api.ToolCallFunction{
						Name:      f.Function.Name,
						Arguments: args,
					},
				}, true
			}
		}
		return api.ToolCall{}, false
	}

	// Try parsing as array first
	if arr := toArray(obj); arr != nil {
		var calls []api.ToolCall
		for _, item := range arr {
			if itemMap, ok := item.(map[string]any); ok {
				var format toolCallFormat
				data, _ := json.Marshal(itemMap)
				if err := json.Unmarshal(data, &format); err == nil {
					if call, ok := makeToolCall(format); ok {
						calls = append(calls, call)
					}
				}
			}
		}
		if len(calls) > 0 {
			return calls, true
		}
	}

	// Try parsing as single object
	var format toolCallFormat
	data, _ := json.Marshal(obj)
	if err := json.Unmarshal(data, &format); err != nil {
		return nil, false
	}

	// Handle xlam format (tool_calls array)
	if len(format.ToolCalls) > 0 {
		var calls []api.ToolCall
		for _, f := range format.ToolCalls {
			if call, ok := makeToolCall(f); ok {
				calls = append(calls, call)
			}
		}
		if len(calls) > 0 {
			return calls, true
		}
	}

	// Try as single tool call
	if call, ok := makeToolCall(format); ok {
		return []api.ToolCall{call}, true
	}

	return nil, false
}

// token, partial, success
func deriveToolToken(s string, prefix string) (string, bool, bool) {
	// There shouldn't be spaces in a tool token
	if len(strings.Fields(s)) > 1 {
		return "", false, false
	}

	if prefix == "[" && len(s) > 1 && s[len(s)-1] == ']' {
		return s, false, true
	} else if prefix == "<" && len(s) > 1 && s[len(s)-1] == '>' {
		return s, false, true
	}
	return "", true, true
}

func parseJSON(s string) ([]api.ToolCall, bool) {
	objs := parseObjects(s)
	tcs := []api.ToolCall{}
	for _, obj := range objs {
		toolCalls, ok := parseJSONToolCalls(obj)
		if ok {
			tcs = append(tcs, toolCalls...)
		}
	}
	if len(tcs) > 0 {
		return tcs, true
	}
	return nil, false
}

// returns tool calls, partial, success
func ParseToolCalls(s string, toolToken *string) ([]api.ToolCall, bool, bool) {
	// [ case can either be JSON, Python or a Tool Token
	s = strings.TrimSpace(s)
	fmt.Printf("ParseToolCallsNew input: %q\n", s)
	if len(s) == 0 {
		return nil, false, false
	}

	if strings.HasPrefix(s, "[") {
		fmt.Println("Found [ prefix")
		// JSON case
		// we do not consider array JSONs as tool calls
		if strings.HasPrefix(s, "[{") {
			fmt.Println("Found [{ prefix - attempting JSON parse")
			// TODO: mark as JSON partial
			if calls, ok := parseJSON(s); ok {
				fmt.Printf("Successfully parsed JSON, found %d calls\n", len(calls))
				return calls, false, true
			}
			return nil, true, true
		}
		// Python Case
		// We just do a full python check here
		fmt.Println("Attempting Python function parse")
		tc, ok := parsePythonFunctionCall(s)
		if ok {
			fmt.Printf("Successfully parsed Python function: %+v\n", tc)
			return tc, false, true
		}
		// Check for partial Python function call
		if strings.Count(s, "(") > strings.Count(s, ")") {
			fmt.Println("Found partial Python function call")
			return nil, true, true
		}
		// Tool Token Case - this is okay if it's a real tool token and we couldn't get from template
		fmt.Println("Attempting to derive tool token")
		if toolToken == nil || *toolToken == "" {
			toolTok, partial, ok := deriveToolToken(s, "[")
			if !ok {
				return nil, false, false
			}
			if partial {
				return nil, true, true
			}
			*toolToken = toolTok
		}
		fmt.Printf("Found tool token: %q\n", *toolToken)
		s = strings.TrimSpace(s[len(*toolToken):])
		fmt.Printf("Recursing with remaining string: %q\n", s)
		if toolCalls, partial, ok := ParseToolCalls(s, toolToken); ok {
			return toolCalls, partial, true
		}
		return nil, true, true
	} else if strings.HasPrefix(s, "{") || strings.HasPrefix(s, "```") {
		fmt.Println("Found { prefix - attempting JSON parse with ", s)
		if calls, ok := parseJSON(s); ok {
			fmt.Printf("Successfully parsed JSON object, found %d calls\n", len(calls))
			return calls, false, true
		}
		fmt.Println("Failed to parse JSON in JSON case")
		// TODO: possible case where it never finishes parsing - then what?
		return nil, true, true
	} else if strings.HasPrefix(s, "<") {
		fmt.Println("Found < prefix - attempting to derive tool token")
		if toolToken == nil || *toolToken == "" {
			toolTok, partial, ok := deriveToolToken(s, "<")
			if !ok {
				return nil, false, false
			}
			if partial {
				return nil, true, true
			}
			*toolToken = toolTok
			fmt.Printf("Found tool token: %q\n", *toolToken)
		}
		fmt.Printf("Found tool token: %q\n", *toolToken)
		s = strings.TrimSpace(s[len(*toolToken):])
		fmt.Printf("Recursing with remaining string: %q\n", s)
		if toolCalls, partial, ok := ParseToolCalls(s, toolToken); ok {
			return toolCalls, partial, true
		}
		return nil, true, true
	} else if strings.Contains(s, "(") || len(strings.Fields(s)) == 1 {
		fmt.Println("Attempting Python function parse")
		tc, ok := parsePythonFunctionCall(s)
		if ok {
			fmt.Printf("Successfully parsed Python function: %+v\n", tc)
			return tc, false, true
		}
		fmt.Printf("Failed to parse Python function: %q, returning partial", s)
		return nil, true, true
	}
	fmt.Println("No successful parse paths found")
	fmt.Printf("failed string: %q\n", s)
	return nil, false, false
}
