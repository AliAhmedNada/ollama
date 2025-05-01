package server

import (
	"testing"
)

func TestToolToken(t *testing.T) {
	cases := []struct {
		input string
		want  string
		ok    bool
	}{
		{
			input: "Hello, world!",
			want:  "",
			ok:    false,
		},
		{
			input: "Hello, world! <tool_call>",
			want:  "<tool_call>",
			ok:    true,
		},
		{
			input: "[tool_call] <tool_call>",
			want:  "[tool_call]",
			ok:    true,
		},
		{
			input: "<tool_call>",
			want:  "<tool_call>",
			ok:    true,
		},
		{
			input: "[tool_call] <",
			want:  "[tool_call]",
			ok:    true,
		},
		{
			input: "> <tool_call>",
			want:  "<tool_call>",
			ok:    true,
		},
		{
			input: "[TOOL_CALL] [",
			want:  "[TOOL_CALL]",
			ok:    true,
		},
	}

	for _, tt := range cases {
		t.Run(tt.input, func(t *testing.T) {
			got, ok := ToolToken(tt.input)
			if got != tt.want {
				t.Errorf("ToolToken(%q) = %q; want %q", tt.input, got, tt.want)
			}
			if ok != tt.ok {
				t.Errorf("ToolToken(%q) = %v; want %v", tt.input, ok, tt.ok)
			}
		})
	}
}
