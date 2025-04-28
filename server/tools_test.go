package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

func readFile(t *testing.T, base, name string) *bytes.Buffer {
	t.Helper()

	bts, err := os.ReadFile(filepath.Join(base, name))
	if err != nil {
		t.Fatal(err)
	}

	return bytes.NewBuffer(bts)
}

func TestExecuteWithTools(t *testing.T) {
	p := filepath.Join("testdata", "tools")
	t1 := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_current_weather",
			Arguments: api.ToolCallFunctionArguments{
				"format":   "fahrenheit",
				"location": "SanFrancisco,CA",
			},
		},
	}
	t2 := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_current_weather",
			Arguments: api.ToolCallFunctionArguments{
				"format":   "celsius",
				"location": "Toronto,Canada",
			},
		},
	}

	cases := []struct {
		model    string
		output   string
		expected []api.ToolCall
		ok       bool
	}{
		{"mistral", `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_curren}]`, []api.ToolCall{}, false},
		{"mistral", `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`, []api.ToolCall{t1, t2}, true},
		{"mistral", `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, `, []api.ToolCall{}, false},
		// ! Not supported anymore
		{"mistral", `I'm not aware of that information. However, I can suggest searching for the weather using the "get_current_weather" function:

		[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`, []api.ToolCall{}, false},
		// ! This is a known regression with the new parsing logic
		{"command-r-plus", "Action: ```json" + `
		[
		    {
		        "tool_name": "get_current_weather",
		        "parameters": {
		            "format": "fahrenheit",
		            "location": "San Francisco, CA"
		        }
		    },
		    {
		        "tool_name": "get_current_weather",
		        "parameters": {
		            "format": "celsius",
		            "location": "Toronto, Canada"
		        }
		    }
		]
		` + "```", []api.ToolCall{}, false},
		// ! This is a known regression with the new parsing logic
		{"firefunction", ` functools[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`, []api.ToolCall{}, false},
		{"llama3-groq-tool-use", `<tool_call>
		{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}
		</tool_call>`, []api.ToolCall{t1}, true},
		{"xlam", `{"tool_calls": [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]}`, []api.ToolCall{t1, t2}, true},
		// ! Not supported anymore
		// {"nemotron", `<toolcall> {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]} </toolcall>`, []api.ToolCall{t1, t2}, true},
		{"nemotron", `<toolcall> {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}, {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}} </toolcall>`, []api.ToolCall{t1, t2}, true},
		{"qwen2.5-coder", `{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}`, []api.ToolCall{t1}, true},
		{"qwen2.5-coder", `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}, {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`, []api.ToolCall{t1, t2}, true},
		{"qwen2.5-coder", " The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.", []api.ToolCall{}, false},
	}

	var tools []api.Tool
	if err := json.Unmarshal(readFile(t, p, "tools.json").Bytes(), &tools); err != nil {
		t.Fatal(err)
	}

	var messages []api.Message
	if err := json.Unmarshal(readFile(t, p, "messages.json").Bytes(), &messages); err != nil {
		t.Fatal(err)
	}

	for _, tt := range cases {
		t.Run(tt.model, func(t *testing.T) {
			tmpl, err := template.Parse(readFile(t, p, fmt.Sprintf("%s.gotmpl", tt.model)).String())
			if err != nil {
				t.Fatal(err)
			}

			t.Run("template", func(t *testing.T) {
				var actual bytes.Buffer
				if err := tmpl.Execute(&actual, template.Values{Tools: tools, Messages: messages}); err != nil {
					t.Fatal(err)
				}

				if diff := cmp.Diff(actual.String(), readFile(t, p, fmt.Sprintf("%s.out", tt.model)).String()); diff != "" {
					t.Errorf("mismatch (-got +want):\n%s", diff)
				}
			})

			t.Run("parse", func(t *testing.T) {
				got := []api.ToolCall{}
				tokens := strings.Fields(tt.output)
				toko := ""
				sb := strings.Builder{}
				for _, tok := range tokens {
					sb.WriteString(tok)
					toolCalls, partial, err := ParseToolCalls(sb.String(), &toko)
					if err != nil {
						t.Fatal(err)
					}
					if partial {
						continue
					}
					got = append(got, toolCalls...)
					sb.Reset()
				}

				if tt.ok {
					if diff := cmp.Diff(got, tt.expected); diff != "" {
						t.Errorf("mismatch (-got +want):\n%s", diff)
					}
				}
			})
		})
	}
}

func TestParseObjects(t *testing.T) {
	tests := []struct {
		input string
		want  []map[string]any
	}{
		{
			input: `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
				{"name": "get_current_weather", "arguments": map[string]any{"format": "celsius", "location": "Toronto, Canada"}},
			},
		},
		{
			input: `<toolcall>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </toolcall>`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
			},
		},
		{
			input: `<toolcall>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </toolcall> <toolcall>{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, ON"}} </toolcall>`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
				{"name": "get_current_weather", "arguments": map[string]any{"format": "celsius", "location": "Toronto, ON"}},
			},
		},
		{
			input: `{"name": "get_current_weather", "arguments": `,
			want:  nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got := parseObjects(tc.input)

			if diff := cmp.Diff(got, tc.want); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
