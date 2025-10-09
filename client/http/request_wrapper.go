package http

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/options"
	"net/http"
)

// RequestWrapper is a struct that encapsulates all necessary data to create and manage an HTTP request for a model inference.
// It includes model details, inputs, outputs, request-specific configurations, and parameters.
type RequestWrapper struct {
	BaseURL      string
	ModelName    string
	ModelVersion string
	Inputs       []base.InferInput
	Outputs      []base.InferOutput
	Options      *options.InferOptions
}

// NewRequestWrapper initializes and returns a new RequestWrapper instance
// with the provided base URL, model details, inputs, outputs, and other configurations.
func NewRequestWrapper(
	baseURL, modelName, modelVersion string,
	inputs []base.InferInput,
	outputs []base.InferOutput,
	opts *options.InferOptions,
) *RequestWrapper {
	if opts == nil {
		opts = &options.InferOptions{
			Headers:                      nil,
			QueryParams:                  nil,
			RequestID:                    nil,
			SequenceID:                   nil,
			SequenceStart:                nil,
			SequenceEnd:                  nil,
			Priority:                     nil,
			Timeout:                      nil,
			RequestCompressionAlgorithm:  nil,
			ResponseCompressionAlgorithm: nil,
			Parameters:                   nil,
		}
	}

	return &RequestWrapper{
		BaseURL:      baseURL,
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs:       inputs,
		Outputs:      outputs,
		Options:      opts,
	}
}

// PrepareRequest prepares the HTTP request for model inference.
// It serializes the inference request body, sets appropriate headers, and constructs the request URI based on model name and version.
func (w *RequestWrapper) PrepareRequest() (*http.Request, error) {
	requestBody, jsonSize, err := w.getInferenceRequest()
	if err != nil {
		return nil, err
	}

	headers := w.prepareHeaders(jsonSize)

	requestURI := fmt.Sprintf("%s/v2/models/%s/infer", w.BaseURL, w.ModelName)
	if w.ModelVersion != "" {
		requestURI = fmt.Sprintf("%s/v2/models/%s/versions/%s/infer", w.BaseURL, w.ModelName, w.ModelVersion)
	}

	req, err := http.NewRequest("POST", requestURI, bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, err
	}

	for key, value := range headers {
		req.Header.Set(key, value)
	}

	return req, nil
}

// getInferenceRequest constructs the inference request payload, serializes it to JSON,
// and appends any binary data associated with the input tensors to the request body.
func (w *RequestWrapper) getInferenceRequest() ([]byte, *int, error) {
	inferRequest := make(map[string]any)
	parameters := make(map[string]any)

	if w.Options.RequestID != nil && *w.Options.RequestID != "" {
		inferRequest["id"] = *w.Options.RequestID
	}
	w.addSequenceParameters(parameters)
	w.addPriorityAndTimeout(parameters)

	inputTensors := w.convertInputsToTensors()
	inferRequest["inputs"] = inputTensors

	if len(w.Outputs) > 0 {
		outputTensors := w.convertOutputsToTensors()
		inferRequest["outputs"] = outputTensors
	} else {
		parameters["binary_data_output"] = true
	}

	if err := w.addCustomParameters(parameters); err != nil {
		return nil, nil, err
	}

	if len(parameters) > 0 {
		inferRequest["parameters"] = parameters
	}

	requestJSON, err := json.Marshal(inferRequest)
	if err != nil {
		return nil, nil, err
	}
	jsonSize := len(requestJSON)

	var requestBody bytes.Buffer
	requestBody.Write(requestJSON)

	for _, inputTensor := range w.Inputs {
		if rawData := inputTensor.GetRawData(); rawData != nil {
			requestBody.Write(rawData)
		}
	}

	if requestBody.Len() == jsonSize {
		return requestBody.Bytes(), nil, nil
	}
	return requestBody.Bytes(), &jsonSize, nil
}

// prepareHeaders prepares the HTTP headers based on request and response compression algorithms,
// and includes the length of the JSON portion of the request if applicable.
func (w *RequestWrapper) prepareHeaders(jsonSize *int) map[string]string {
	headers := make(map[string]string)
	if w.Options.RequestCompressionAlgorithm != nil && *w.Options.RequestCompressionAlgorithm != "" {
		switch *w.Options.RequestCompressionAlgorithm {
		case "gzip":
			headers["Content-Encoding"] = "gzip"
		case "deflate":
			headers["Content-Encoding"] = "deflate"
		}
	}

	if w.Options.ResponseCompressionAlgorithm != nil && *w.Options.ResponseCompressionAlgorithm != "" {
		headers["Accept-Encoding"] = *w.Options.ResponseCompressionAlgorithm
	}

	if jsonSize != nil {
		headers["Inference-Header-Content-Length"] = fmt.Sprintf("%d", *jsonSize)
	}

	return headers
}

// addSequenceParameters adds sequence-related parameters to the inference request if sequence details (ID, start, end) are provided.
func (w *RequestWrapper) addSequenceParameters(parameters map[string]any) {
	if w.Options.SequenceID != nil {
		parameters["sequence_id"] = *w.Options.SequenceID
		if w.Options.SequenceStart != nil {
			parameters["sequence_start"] = *w.Options.SequenceStart
		}
		if w.Options.SequenceEnd != nil {
			parameters["sequence_end"] = *w.Options.SequenceEnd
		}
	}
}

// addPriorityAndTimeout adds priority and timeout parameters to the inference request if specified.
func (w *RequestWrapper) addPriorityAndTimeout(parameters map[string]any) {
	if w.Options.Priority != nil {
		parameters["priority"] = *w.Options.Priority
	}
	if w.Options.Timeout != nil {
		parameters["timeout"] = *w.Options.Timeout
	}
}

// addCustomParameters adds any custom parameters to the inference request,
// ensuring that no reserved parameters are overwritten, and returns an error if any reserved parameters are used.
func (w *RequestWrapper) addCustomParameters(parameters map[string]any) error {
	for key, value := range w.Options.Parameters {
		switch key {
		case "sequence_id", "sequence_start", "sequence_end", "priority", "binary_data_output":
			return fmt.Errorf("parameter %q is a reserved parameter and cannot be specified", key)
		default:
			parameters[key] = value
		}
	}
	return nil
}

// convertInputsToTensors converts the input base.InferInput instances to a format suitable for the inference request payload.
func (w *RequestWrapper) convertInputsToTensors() []map[string]any {
	inputTensors := make([]map[string]any, len(w.Inputs))
	for i, input := range w.Inputs {
		inputTensors[i] = input.GetTensor().(map[string]any)
	}
	return inputTensors
}

// convertOutputsToTensors converts the output base.InferOutput instances to a format suitable for the inference request payload.
func (w *RequestWrapper) convertOutputsToTensors() []map[string]any {
	outputTensors := make([]map[string]any, len(w.Outputs))
	for i, output := range w.Outputs {
		outputTensors[i] = output.GetTensor().(map[string]any)
	}
	return outputTensors
}
