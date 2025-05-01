package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/ollama/ollama/api"
)

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

// called after finding a tool token
func simpleParse(s string) ([]api.ToolCall, bool, error) {
	if strings.HasPrefix(s, "[") {
		fmt.Println("Found [ prefix")
		// JSON case
		// we do not consider array JSONs as tool calls
		if strings.HasPrefix(s, "[{") {
			fmt.Println("Found [{ prefix - attempting JSON parse")
			// TODO: mark as JSON partial
			if calls, ok := parseJSON(s); ok {
				fmt.Printf("Successfully parsed JSON, found %d calls\n", len(calls))
				return calls, false, nil
			}
			return nil, true, nil
		}
	} else if strings.HasPrefix(s, "{") || strings.HasPrefix(s, "```") {
		fmt.Println("Found { prefix - attempting JSON parse with ", s)
		if calls, ok := parseJSON(s); ok {
			fmt.Printf("Successfully parsed JSON object, found %d calls\n", len(calls))
			return calls, false, nil
		}
		fmt.Println("Failed to parse JSON in JSON case")
		// It is possible that the JSON never finishes - in which case it should be sent back on done as content
		return nil, true, nil
	}

	fmt.Println("No successful parse paths found")
	fmt.Printf("failed string: %q\n", s)
	fmt.Println("returning partial")
	return nil, false, fmt.Errorf("no successful parse paths found")
}

// returns tool calls, partial, success
// ParseToolCalls attempts to parse tool calls from a string, handling various formats
// Returns:
// - []api.ToolCall: Any successfully parsed tool calls
// - bool: Whether this is a partial parse that needs more input
// - error: Any error encountered during parsing, or nil if successful
func ParseToolCalls(s string, toolToken *string) ([]api.ToolCall, bool, error) {
	// if toolToken == nil {
	// 	toolToken = new(string)
	// }
	// [ case can either be JSON, Python or a Tool Token
	s = strings.TrimSpace(s)
	fmt.Printf("ParseToolCallsNew input: %q\n", s)
	if len(s) == 0 {
		return nil, false, fmt.Errorf("empty input string")
	}
	if *toolToken != "" {
		if strings.HasPrefix(s, *toolToken) {
			s = strings.TrimSpace(s[len(*toolToken):])
			fmt.Printf("Recursing with remaining string: %q\n", s)
			tc, _, err := simpleParse(s)
			// TODO: clean this up
			if err != nil {
				return nil, true, nil
			}
			if len(tc) == 0 {
				fmt.Println("No tool calls found in remaining string, partial")
				return nil, true, nil
			}
			return tc, false, nil
			// TODO: this has to be common for all tool tokens
		} else if strings.HasSuffix(s, (*toolToken)[1:]) {
			fmt.Println("Found tool token suffix")
			// TODO: end of special token case
			fmt.Println("Found end of special token")
			// Dummy flag
			tc := api.ToolCall{
				Function: api.ToolCallFunction{
					Name: *toolToken,
				},
			}
			return []api.ToolCall{tc}, true, nil
		} else {
			return nil, false, fmt.Errorf("tool token not found in input")
		}
	}
	return simpleParse(s)
}
