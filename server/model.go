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

// Get tool call token from model template
func (m *Model) TemplateToolToken() (string, string, bool) {
	// Try to detect the tool call format from the model's template
	tmpl := m.Template.Subtree(func(n parse.Node) bool {
		if t, ok := n.(*parse.RangeNode); ok {
			return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
		}
		return false
	})

	if tmpl != nil {
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
				// Extract the special token between < and >
				start := strings.Index(output, "<")
				end := strings.Index(output, ">")
				if start >= 0 && end > start {
					token := output[start : end+1]
					return output, token, true
				}
			} else if strings.Contains(output, "[") {
				// Check if it's a tool call token rather than JSON array
				start := strings.Index(output, "[")
				end := strings.Index(output, "]")
				if start >= 0 && end > start {
					token := output[start : end+1]
					// There shouldn't be spaces in a special token
					if len(strings.Fields(token)) > 1 {
						return "", "", false
					}

					// Only consider it a token if it's not valid JSON
					var jsonTest any
					if err := json.Unmarshal([]byte(token), &jsonTest); err != nil {
						return output, token, true
					}
				}
			}
		}
	}
	return "", "", false
}
