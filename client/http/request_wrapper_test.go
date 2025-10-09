package http

import (
	"encoding/json"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/options"
	"net/http"
	"reflect"
	"testing"
)

func TestNewRequestWrapper(t *testing.T) {
	baseURL := "http://localhost:8000"
	modelName := "test_model"
	modelVersion := "1"
	inputs := []base.InferInput{}
	outputs := []base.InferOutput{}
	requestID := new(string)
	*requestID = "test_request_id"
	sequenceID := new(int)
	*sequenceID = 42
	sequenceStart := new(bool)
	*sequenceStart = true
	sequenceEnd := new(bool)
	*sequenceEnd = false
	priority := new(int)
	*priority = 1
	timeout := new(int)
	*timeout = 1000
	requestCompressionAlgorithm := new(string)
	*requestCompressionAlgorithm = "gzip"
	responseCompressionAlgorithm := new(string)
	*responseCompressionAlgorithm = "deflate"
	parameters := map[string]any{"custom_param": "value"}
	opts := &options.InferOptions{
		Headers:                      nil,
		QueryParams:                  nil,
		RequestID:                    requestID,
		SequenceID:                   sequenceID,
		SequenceStart:                sequenceStart,
		SequenceEnd:                  sequenceEnd,
		Priority:                     priority,
		Timeout:                      timeout,
		RequestCompressionAlgorithm:  requestCompressionAlgorithm,
		ResponseCompressionAlgorithm: responseCompressionAlgorithm,
		Parameters:                   parameters,
	}

	wrapper := NewRequestWrapper(
		baseURL,
		modelName,
		modelVersion,
		inputs,
		outputs,
		opts,
	)

	if wrapper.BaseURL != baseURL {
		t.Errorf("Expected BaseURL %s, got %s", baseURL, wrapper.BaseURL)
	}
	if wrapper.ModelName != modelName {
		t.Errorf("Expected ModelName %s, got %s", modelName, wrapper.ModelName)
	}
	if wrapper.ModelVersion != modelVersion {
		t.Errorf("Expected ModelVersion %s, got %s", modelVersion, wrapper.ModelVersion)
	}
	if wrapper.Options.RequestID == nil || *wrapper.Options.RequestID != *requestID {
		t.Errorf("Expected RequestID %s, got %v", *requestID, wrapper.Options.RequestID)
	}
	if wrapper.Options.SequenceID == nil || *wrapper.Options.SequenceID != *sequenceID {
		t.Errorf("Expected SequenceID %d, got %v", *sequenceID, wrapper.Options.SequenceID)
	}
	if wrapper.Options.SequenceStart == nil || *wrapper.Options.SequenceStart != *sequenceStart {
		t.Errorf("Expected SequenceStart %v, got %v", *sequenceStart, wrapper.Options.SequenceStart)
	}
	if wrapper.Options.SequenceEnd == nil || *wrapper.Options.SequenceEnd != *sequenceEnd {
		t.Errorf("Expected SequenceEnd %v, got %v", *sequenceEnd, wrapper.Options.SequenceEnd)
	}
	if wrapper.Options.Priority == nil || *wrapper.Options.Priority != *priority {
		t.Errorf("Expected Priority %d, got %v", *priority, wrapper.Options.Priority)
	}
	if wrapper.Options.Timeout == nil || *wrapper.Options.Timeout != *timeout {
		t.Errorf("Expected Timeout %d, got %v", *timeout, wrapper.Options.Timeout)
	}
	if wrapper.Options.RequestCompressionAlgorithm == nil || *wrapper.Options.RequestCompressionAlgorithm != *requestCompressionAlgorithm {
		t.Errorf("Expected RequestCompressionAlgorithm %s, got %v", *requestCompressionAlgorithm, wrapper.Options.RequestCompressionAlgorithm)
	}
	if wrapper.Options.ResponseCompressionAlgorithm == nil || *wrapper.Options.ResponseCompressionAlgorithm != *responseCompressionAlgorithm {
		t.Errorf("Expected ResponseCompressionAlgorithm %s, got %v", *responseCompressionAlgorithm, wrapper.Options.ResponseCompressionAlgorithm)
	}
	if !reflect.DeepEqual(wrapper.Options.Parameters, parameters) {
		t.Errorf("Expected Parameters %v, got %v", parameters, wrapper.Options.Parameters)
	}
	if wrapper.Options != opts {
		t.Errorf("Expected Options %v, got %v", opts, wrapper.Options)
	}
}

func TestPrepareRequest(t *testing.T) {
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		[]base.InferInput{},
		[]base.InferOutput{},
		nil,
	)

	req, err := wrapper.PrepareRequest()
	if err != nil {
		t.Fatalf("PrepareRequest returned error: %v", err)
	}

	expectedURL := "http://localhost:8000/v2/models/test_model/infer"
	if req.URL.String() != expectedURL {
		t.Errorf("Expected URL %s, got %s", expectedURL, req.URL.String())
	}
	if req.Method != http.MethodPost {
		t.Errorf("Expected method POST, got %s", req.Method)
	}
}

func TestPrepareRequestWithVersion(t *testing.T) {
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"2",
		[]base.InferInput{},
		[]base.InferOutput{},
		nil,
	)

	req, err := wrapper.PrepareRequest()
	if err != nil {
		t.Fatalf("PrepareRequest returned error: %v", err)
	}

	expectedURL := "http://localhost:8000/v2/models/test_model/versions/2/infer"
	if req.URL.String() != expectedURL {
		t.Errorf("Expected URL %s, got %s", expectedURL, req.URL.String())
	}
}

func TestGetInferenceRequest(t *testing.T) {
	inputData := []int32{1, 2, 3, 4}
	input := NewInferInput("input0", "INT32", []int64{2, 2}, nil)
	input.SetData(inputData, true)
	requestId := "test-id"
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		[]base.InferInput{input},
		[]base.InferOutput{},
		&options.InferOptions{
			RequestID: &requestId,
		},
	)

	requestBody, jsonSize, err := wrapper.getInferenceRequest()
	if err != nil {
		t.Fatalf("getInferenceRequest returned error: %v", err)
	}

	if jsonSize == nil {
		t.Error("Expected jsonSize to be non-nil")
	}

	expectedSize := len(requestBody) - len(input.GetRawData())
	if *jsonSize != expectedSize {
		t.Errorf("Expected jsonSize %d, got %d", expectedSize, *jsonSize)
	}

	var inferRequest map[string]any
	err = json.Unmarshal(requestBody[:*jsonSize], &inferRequest)
	if err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	if _, ok := inferRequest["inputs"]; !ok {
		t.Error("Expected 'inputs' key in inferRequest")
	}
}

func TestGetInferenceRequestError(t *testing.T) {
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		nil,
		nil,
		&options.InferOptions{
			Parameters: map[string]any{
				"sequence_id": 123,
			},
		},
	)

	_, _, err := wrapper.getInferenceRequest()
	if err == nil {
		t.Fatal("Expected error from getInferenceRequest, got nil")
	}
}

func TestPrepareHeaders(t *testing.T) {
	jsonSize := 100
	gzipAlg := "gzip"
	deflateAlg := "deflate"
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		nil,
		nil,
		&options.InferOptions{
			RequestCompressionAlgorithm:  &gzipAlg,
			ResponseCompressionAlgorithm: &deflateAlg,
		},
	)
	headers := wrapper.prepareHeaders(&jsonSize)
	if headers["Inference-Header-Content-Length"] != "100" {
		t.Errorf("Expected Inference-Header-Content-Length '100', got %s", headers["Inference-Header-Content-Length"])
	}
	if headers["Content-Encoding"] != "gzip" {
		t.Errorf("Expected Content-Encoding 'gzip', got %s", headers["Content-Encoding"])
	}
	if headers["Accept-Encoding"] != "deflate" {
		t.Errorf("Expected Accept-Encoding 'deflate', got %s", headers["Accept-Encoding"])
	}
}

func TestPrepareHeadersNoCompression(t *testing.T) {
	jsonSize := 100
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		nil,
		nil,
		nil,
	)
	headers := wrapper.prepareHeaders(&jsonSize)
	if headers["Inference-Header-Content-Length"] != "100" {
		t.Errorf("Expected Inference-Header-Content-Length '100', got %s", headers["Inference-Header-Content-Length"])
	}
	if _, ok := headers["Content-Encoding"]; ok {
		t.Errorf("Did not expect Content-Encoding header, got %s", headers["Content-Encoding"])
	}
	if _, ok := headers["Accept-Encoding"]; ok {
		t.Errorf("Did not expect Accept-Encoding header, got %s", headers["Accept-Encoding"])
	}
}

func TestAddSequenceParameters(t *testing.T) {
	parameters := make(map[string]any)
	sequenceID := 42
	sequenceStart := true
	sequenceEnd := false
	wrapper := NewRequestWrapper(
		"",
		"",
		"",
		nil,
		nil,
		&options.InferOptions{
			SequenceID:    &sequenceID,
			SequenceStart: &sequenceStart,
			SequenceEnd:   &sequenceEnd,
		},
	)
	wrapper.addSequenceParameters(parameters)
	if val, ok := parameters["sequence_id"]; !ok || val != sequenceID {
		t.Errorf("Expected sequence_id %d, got %v", sequenceID, val)
	}
	if val, ok := parameters["sequence_start"]; !ok || val != sequenceStart {
		t.Errorf("Expected sequence_start %v, got %v", sequenceStart, val)
	}
	if val, ok := parameters["sequence_end"]; !ok || val != sequenceEnd {
		t.Errorf("Expected sequence_end %v, got %v", sequenceEnd, val)
	}
}

func TestAddPriorityAndTimeout(t *testing.T) {
	parameters := make(map[string]any)
	priority := 1
	timeout := 1000
	wrapper := NewRequestWrapper(
		"",
		"",
		"",
		nil,
		nil,
		&options.InferOptions{
			Priority: &priority,
			Timeout:  &timeout,
		},
	)
	wrapper.addPriorityAndTimeout(parameters)
	if val, ok := parameters["priority"]; !ok || val != priority {
		t.Errorf("Expected priority %d, got %v", priority, val)
	}
	if val, ok := parameters["timeout"]; !ok || val != timeout {
		t.Errorf("Expected timeout %d, got %v", timeout, val)
	}
}

func TestAddCustomParameters(t *testing.T) {
	parameters := make(map[string]any)
	wrapper := NewRequestWrapper(
		"",
		"",
		"",
		nil,
		nil,
		&options.InferOptions{
			Parameters: map[string]any{
				"custom_param": "value",
			},
		},
	)
	err := wrapper.addCustomParameters(parameters)
	if err != nil {
		t.Fatalf("addCustomParameters returned error: %v", err)
	}
	if val, ok := parameters["custom_param"]; !ok || val != "value" {
		t.Errorf("Expected custom_param to be 'value', got %v", val)
	}
}

func TestAddCustomParametersReservedKey(t *testing.T) {
	reservedKeys := []string{"sequence_id", "sequence_start", "sequence_end", "priority", "binary_data_output"}
	for _, key := range reservedKeys {
		parameters := make(map[string]any)
		wrapper := NewRequestWrapper(
			"",
			"",
			"",
			nil,
			nil,
			&options.InferOptions{
				Parameters: map[string]any{
					key: 123,
				},
			},
		)
		err := wrapper.addCustomParameters(parameters)
		if err == nil {
			t.Errorf("Expected error for reserved parameter %q, got nil", key)
		}
	}
}

func TestConvertInputsToTensors(t *testing.T) {
	input := NewInferInput("input0", "FP32", []int64{1, 3}, map[string]any{"test": "test"})
	wrapper := NewRequestWrapper(
		"",
		"",
		"",
		[]base.InferInput{input},
		nil,
		nil,
	)
	tensors := wrapper.convertInputsToTensors()
	if len(tensors) != 1 {
		t.Errorf("Expected 1 tensor, got %d", len(tensors))
	}
	expectedTensor := map[string]any{
		"name":       "input0",
		"shape":      []int64{1, 3},
		"datatype":   "FP32",
		"parameters": map[string]any{"test": "test"},
	}
	if !reflect.DeepEqual(tensors[0], expectedTensor) {
		t.Errorf("Expected tensor %v, got %v", expectedTensor, tensors[0])
	}
}

func TestConvertOutputsToTensors(t *testing.T) {
	output := NewInferOutput("output0", nil)
	wrapper := NewRequestWrapper(
		"",
		"",
		"",
		nil,
		[]base.InferOutput{output},
		nil,
	)
	tensors := wrapper.convertOutputsToTensors()
	if len(tensors) != 1 {
		t.Errorf("Expected 1 tensor, got %d", len(tensors))
	}
	expectedTensor := map[string]any{
		"name":       "output0",
		"parameters": map[string]any{},
	}
	if !reflect.DeepEqual(tensors[0], expectedTensor) {
		t.Errorf("Expected tensor %v, got %v", expectedTensor, tensors[0])
	}
}

func TestPrepareHeadersNilJsonSize(t *testing.T) {
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		nil,
		nil,
		nil,
	)
	headers := wrapper.prepareHeaders(nil)
	if _, ok := headers["Inference-Header-Content-Length"]; ok {
		t.Errorf("Did not expect Inference-Header-Content-Length header, got %s", headers["Inference-Header-Content-Length"])
	}
}

func TestGetInferenceRequestNoInputs(t *testing.T) {
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		[]base.InferInput{},
		nil,
		nil,
	)

	requestBody, jsonSize, err := wrapper.getInferenceRequest()
	if err != nil {
		t.Fatalf("getInferenceRequest returned error: %v", err)
	}

	if jsonSize != nil {
		t.Error("Expected jsonSize to be non-nil")
	}

	var inferRequest map[string]any
	err = json.Unmarshal(requestBody, &inferRequest)
	if err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	if inputs, ok := inferRequest["inputs"]; !ok || len(inputs.([]any)) != 0 {
		t.Errorf("Expected 'inputs' to be empty, got %v", inputs)
	}
}

func TestGetInferenceRequestWithOutputs(t *testing.T) {
	inputData := []int32{1, 2, 3, 4}
	input := NewInferInput("input0", "INT32", []int64{2, 2}, nil)
	input.SetData(inputData, true)
	output := NewInferOutput("output0", nil)
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		[]base.InferInput{input},
		[]base.InferOutput{output},
		nil,
	)

	requestBody, jsonSize, err := wrapper.getInferenceRequest()
	if err != nil {
		t.Fatalf("getInferenceRequest returned error: %v", err)
	}

	if jsonSize == nil {
		t.Error("Expected jsonSize to be non-nil")
	}

	expectedSize := len(requestBody) - len(input.GetRawData())
	if *jsonSize != expectedSize {
		t.Errorf("Expected jsonSize %d, got %d", expectedSize, *jsonSize)
	}

	var inferRequest map[string]any
	err = json.Unmarshal(requestBody[:*jsonSize], &inferRequest)
	if err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	if outputs, ok := inferRequest["outputs"]; !ok || len(outputs.([]any)) != 1 {
		t.Errorf("Expected 'outputs' with 1 item, got %v", outputs)
	}
}

func TestPrepareRequestError(t *testing.T) {
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		nil,
		nil,
		&options.InferOptions{
			Parameters: map[string]any{
				"sequence_id": 123,
			},
		},
	)

	_, err := wrapper.PrepareRequest()
	if err == nil {
		t.Fatal("Expected error from PrepareRequest, got nil")
	}
}

func TestGetInferenceRequestNoRawData(t *testing.T) {
	input := NewInferInput("input0", "INT32", []int64{2, 2}, nil)
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		[]base.InferInput{input},
		nil,
		nil,
	)

	requestBody, jsonSize, err := wrapper.getInferenceRequest()
	if err != nil {
		t.Fatalf("getInferenceRequest returned error: %v", err)
	}

	if jsonSize != nil {
		t.Errorf("Expected jsonSize to be nil, got %v", *jsonSize)
	}

	expectedSize := len(requestBody)
	if expectedSize == 0 {
		t.Error("Expected non-empty requestBody")
	}
}

func TestPrepareHeadersInvalidAlgorithm(t *testing.T) {
	invalidAlg := "invalid"
	wrapper := NewRequestWrapper(
		"",
		"",
		"",
		nil,
		nil,
		&options.InferOptions{
			RequestCompressionAlgorithm: &invalidAlg,
		},
	)
	headers := wrapper.prepareHeaders(nil)
	if _, ok := headers["Content-Encoding"]; ok {
		t.Errorf("Did not expect Content-Encoding header, got %s", headers["Content-Encoding"])
	}
}

func TestPrepareRequest_NewRequestError(t *testing.T) {
	wrapper := NewRequestWrapper(
		"http://[::1]:namedport",
		"test_model",
		"",
		[]base.InferInput{},
		[]base.InferOutput{},
		nil,
	)

	_, err := wrapper.PrepareRequest()
	if err == nil {
		t.Fatal("Expected error from PrepareRequest due to invalid URL, got nil")
	}
}

func TestPrepareRequest_SetHeaders(t *testing.T) {
	requestID := new(string)
	*requestID = "test_request_id"
	sequenceID := new(int)
	*sequenceID = 42
	sequenceStart := new(bool)
	*sequenceStart = true
	sequenceEnd := new(bool)
	*sequenceEnd = false
	priority := new(int)
	*priority = 1
	timeout := new(int)
	*timeout = 1000
	requestCompressionAlgorithm := new(string)
	*requestCompressionAlgorithm = "gzip"
	responseCompressionAlgorithm := new(string)
	*responseCompressionAlgorithm = "deflate"
	parameters := map[string]any{"custom_param": "value"}

	opts := &options.InferOptions{
		Headers:                      nil,
		QueryParams:                  nil,
		RequestID:                    requestID,
		SequenceID:                   sequenceID,
		SequenceStart:                sequenceStart,
		SequenceEnd:                  sequenceEnd,
		Priority:                     priority,
		Timeout:                      timeout,
		RequestCompressionAlgorithm:  requestCompressionAlgorithm,
		ResponseCompressionAlgorithm: responseCompressionAlgorithm,
		Parameters:                   parameters,
	}

	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		[]base.InferInput{},
		[]base.InferOutput{},
		opts,
	)

	req, err := wrapper.PrepareRequest()
	if err != nil {
		t.Fatalf("PrepareRequest returned error: %v", err)
	}

	expectedHeaders := map[string]string{
		"Content-Encoding": "gzip",
		"Accept-Encoding":  "deflate",
	}

	for key, expectedValue := range expectedHeaders {
		if req.Header.Get(key) != expectedValue {
			t.Errorf("Expected header %s to be %s, got %s", key, expectedValue, req.Header.Get(key))
		}
	}
}

func TestGetInferenceRequest_JSONMarshalError(t *testing.T) {
	invalidValue := make(chan int)
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		nil,
		nil,
		&options.InferOptions{
			Parameters: map[string]any{
				"invalid_param": invalidValue,
			},
		},
	)

	_, _, err := wrapper.getInferenceRequest()
	if err == nil {
		t.Fatal("Expected error from getInferenceRequest due to json.Marshal failure, got nil")
	}
}

func TestPrepareHeaders_DeflateCompression(t *testing.T) {
	jsonSize := 100
	deflateAlg := "deflate"
	wrapper := NewRequestWrapper(
		"http://localhost:8000",
		"test_model",
		"",
		nil,
		nil,
		&options.InferOptions{
			RequestCompressionAlgorithm: &deflateAlg,
		},
	)
	headers := wrapper.prepareHeaders(&jsonSize)
	if headers["Content-Encoding"] != "deflate" {
		t.Errorf("Expected Content-Encoding 'deflate', got %s", headers["Content-Encoding"])
	}
}
