package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strings"
	gotmpl "text/template"
	"text/template/parse"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

var intermediateBlobs map[string]string = make(map[string]string)

type layerGGML struct {
	Layer
	*ggml.GGML
}

func parseFromModel(ctx context.Context, name model.Name, fn func(api.ProgressResponse)) (layers []*layerGGML, err error) {
	m, err := ParseNamedManifest(name)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if err := PullModel(ctx, name.String(), &registryOptions{}, fn); err != nil {
			return nil, err
		}

		m, err = ParseNamedManifest(name)
		if err != nil {
			return nil, err
		}
	case err != nil:
		return nil, err
	}

	for _, layer := range m.Layers {
		layer, err := NewLayerFromLayer(layer.Digest, layer.MediaType, name.DisplayShortest())
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model",
			"application/vnd.ollama.image.projector",
			"application/vnd.ollama.image.adapter":
			blobpath, err := GetBlobsPath(layer.Digest)
			if err != nil {
				return nil, err
			}

			blob, err := os.Open(blobpath)
			if err != nil {
				return nil, err
			}
			defer blob.Close()

			f, _, err := ggml.Decode(blob, 1024)
			if err != nil {
				return nil, err
			}

			layers = append(layers, &layerGGML{layer, f})
		default:
			layers = append(layers, &layerGGML{layer, nil})
		}
	}

	return layers, nil
}

func detectChatTemplate(layers []*layerGGML) ([]*layerGGML, error) {
	for _, layer := range layers {
		if s := layer.GGML.KV().ChatTemplate(); s != "" {
			if t, err := template.Named(s); err != nil {
				slog.Debug("template detection", "error", err, "template", s)
			} else {
				layer, err := NewLayer(t.Reader(), "application/vnd.ollama.image.template")
				if err != nil {
					return nil, err
				}

				layer.status = fmt.Sprintf("using autodetected template %s", t.Name)
				layers = append(layers, &layerGGML{layer, nil})

				if t.Parameters != nil {
					var b bytes.Buffer
					if err := json.NewEncoder(&b).Encode(t.Parameters); err != nil {
						return nil, err
					}

					layer, err := NewLayer(&b, "application/vnd.ollama.image.params")
					if err != nil {
						return nil, err
					}

					layers = append(layers, &layerGGML{layer, nil})
				}
			}
		}
	}

	return layers, nil
}

func detectContentType(r io.Reader) (string, error) {
	var b bytes.Buffer
	if _, err := io.Copy(&b, r); err != nil {
		return "", err
	}

	if contentType := ggml.DetectContentType(b.Bytes()); contentType != "" {
		return contentType, nil
	}

	if contentType := http.DetectContentType(b.Bytes()); contentType != "application/octet-stream" {
		return contentType, nil
	}

	return "unknown", nil
}

// textAfterToolCalls finds the immediate following text after any IfNode containing ".ToolCalls"
func textAfterToolCalls(tmpl *gotmpl.Template) (string, bool) {
	if tmpl == nil || tmpl.Tree == nil {
		return "", false
	}

	var result string
	var found bool

	var walk func(nodes []parse.Node)
	walk = func(nodes []parse.Node) {
		for _, node := range nodes {
			if found {
				return
			}

			switch n := node.(type) {
			case *parse.IfNode:
				if nodeContainsToolCalls(n) {
					// Collect immediate TextNode(s) at start of IfNode's list
					var sb strings.Builder
					for _, innerNode := range n.List.Nodes {
						if tn, ok := innerNode.(*parse.TextNode); ok {
							sb.Write(tn.Text)
						} else {
							// Stop at first non-text node
							break
						}
					}
					result = sb.String()
					found = true
					return
				}
				// Recurse into child nodes
				walk(n.List.Nodes)
				if n.ElseList != nil {
					walk(n.ElseList.Nodes)
				}
			case *parse.ListNode:
				walk(n.Nodes)
			case *parse.RangeNode:
				walk(n.List.Nodes)
				if n.ElseList != nil {
					walk(n.ElseList.Nodes)
				}
			case *parse.WithNode:
				walk(n.List.Nodes)
				if n.ElseList != nil {
					walk(n.ElseList.Nodes)
				}
			default:
				// Continue to next node
				continue
			}

			if found {
				return
			}
		}
	}

	walk(tmpl.Tree.Root.Nodes)
	return result, found
}

// Helper to detect if a node's condition includes ".ToolCalls"
func nodeContainsToolCalls(n *parse.IfNode) bool {
	for _, cmd := range n.Pipe.Cmds {
		for _, arg := range cmd.Args {
			if field, ok := arg.(*parse.FieldNode); ok {
				for _, ident := range field.Ident {
					if ident == "ToolCalls" {
						return true
					}
				}
			}
		}
	}
	return false
}

func ToolToken(found string) (string, bool) {
	if found == "" {
		return "", false
	}
	start := -1
	end := -1
	for i, r := range found {
		if r == '<' || r == '[' {
			start = i
		}
		if (r == '>' || r == ']') && start != -1 {
			end = i
			break
		}
	}
	if start == -1 || end == -1 {
		return "", false
	}
	return found[start : end+1], true
}

// Get tool call token from model template
func (m *Model) TemplateToolToken() (string, string, bool) {
	// Try to detect the tool call format from the model's template
	slog.Debug("attempting to detect tool call format from template")
	tmpl := m.Template.Subtree(func(n parse.Node) bool {
		if t, ok := n.(*parse.RangeNode); ok {
			return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
		}
		return false
	})

	if tmpl != nil {
		slog.Debug("found tool calls template node")
		// Execute template with test data to see the format
		var b bytes.Buffer
		if err := tmpl.Execute(&b, map[string][]api.ToolCall{
			"ToolCalls": {
				{
					Function: api.ToolCallFunction{
						Name: "function_name",
						Arguments: api.ToolCallFunctionArguments{
							"argument1": "value1",
						},
					},
				},
			},
		}); err == nil {
			// Look for special tokens in the template output
			output := strings.TrimSpace(b.String())
			slog.Debug("tool call template output", "output", output)
			if strings.Contains(output, "<") {
				slog.Debug("found < token in output")
				// Extract the special token between < and >
				start := strings.Index(output, "<")
				end := strings.Index(output, ">")
				if start >= 0 && end > start {
					token := output[start : end+1]
					slog.Debug("extracted token", "token", token)
					return output, token, true
				}
			} else if strings.Contains(output, "[") {
				slog.Debug("found [ token in output")
				// Check if it's a tool call token rather than JSON array
				start := strings.Index(output, "[")
				end := strings.Index(output, "]")
				if start >= 0 && end > start {
					token := output[start : end+1]
					slog.Debug("potential token", "token", token)
					// There shouldn't be spaces in a special token
					if len(strings.Fields(token)) > 1 {
						slog.Debug("token contains spaces, not a valid token")
						return "", "", false
					}

					// Only consider it a token if it's not valid JSON
					var jsonTest any
					if err := json.Unmarshal([]byte(token), &jsonTest); err != nil {
						slog.Debug("token is not valid JSON, treating as tool token")
						return output, token, true
					}
					slog.Debug("token is valid JSON, not a tool token")
				}
			}
		} else {
			slog.Debug("failed to execute template", "error", err)
		}
	} else {
		slog.Debug("no tool calls template node found")
	}
	return "", "", false
}
