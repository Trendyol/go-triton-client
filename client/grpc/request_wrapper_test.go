package grpc

import (
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/options"
	"reflect"
	"testing"
)

func TestNewRequestWrapper(t *testing.T) {
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
		modelName,
		modelVersion,
		inputs,
		outputs,
		opts,
	)

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

func TestPrepareRequest_Success(t *testing.T) {
	inputData := []int32{1, 2, 3, 4}
	input := NewInferInput("input0", "INT32", []int64{2, 2}, nil)
	input.SetData(inputData, true)
	output := NewInferOutput("output0", nil)
	wrapper := NewRequestWrapper(
		"test_model",
		"",
		[]base.InferInput{input},
		[]base.InferOutput{output},
		nil,
	)

	request, err := wrapper.PrepareRequest()
	if err != nil {
		t.Fatalf("PrepareRequest returned error: %v", err)
	}

	if request.ModelName != "test_model" {
		t.Errorf("Expected ModelName 'test_model', got %s", request.ModelName)
	}
	if len(request.Inputs) != 1 {
		t.Errorf("Expected 1 input, got %d", len(request.Inputs))
	}
	if len(request.RawInputContents) != 1 {
		t.Errorf("Expected 1 RawInputContent, got %d", len(request.RawInputContents))
	}
	if len(request.Outputs) != 1 {
		t.Errorf("Expected 1 output, got %d", len(request.Outputs))
	}
}

func TestPrepareRequest_Error(t *testing.T) {
	parameters := map[string]any{
		"sequence_id": 123,
	}
	wrapper := NewRequestWrapper(
		"test_model",
		"",
		nil,
		nil,
		&options.InferOptions{
			Parameters: parameters,
		},
	)

	_, err := wrapper.PrepareRequest()
	if err == nil {
		t.Fatal("Expected error from PrepareRequest due to reserved parameter, got nil")
	}
}

func TestGetInferenceRequest_NoOutputs(t *testing.T) {
	inputData := []int32{1, 2, 3, 4}
	input := NewInferInput("input0", "INT32", []int64{2, 2}, nil)
	input.SetData(inputData, true)
	requestId := "test-id"
	wrapper := NewRequestWrapper(
		"test_model",
		"",
		[]base.InferInput{input},
		nil,
		&options.InferOptions{
			RequestID: &requestId,
		},
	)

	request, err := wrapper.getInferenceRequest()
	if err != nil {
		t.Fatalf("getInferenceRequest returned error: %v", err)
	}
	if param, ok := request.Parameters["binary_data_output"]; !ok || param.GetBoolParam() != true {
		t.Errorf("Expected binary_data_output true, got %v", param)
	}
}

func TestAddSequenceParameters(t *testing.T) {
	parameters := make(map[string]*grpc_generated_v2.InferParameter)
	sequenceID := 42
	sequenceStart := true
	sequenceEnd := false
	wrapper := NewRequestWrapper(
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
	if param, ok := parameters["sequence_id"]; !ok || param.GetInt64Param() != int64(sequenceID) {
		t.Errorf("Expected sequence_id %d, got %v", sequenceID, param)
	}
	if param, ok := parameters["sequence_start"]; !ok || param.GetBoolParam() != sequenceStart {
		t.Errorf("Expected sequence_start %v, got %v", sequenceStart, param)
	}
	if param, ok := parameters["sequence_end"]; !ok || param.GetBoolParam() != sequenceEnd {
		t.Errorf("Expected sequence_end %v, got %v", sequenceEnd, param)
	}
}

func TestAddSequenceParameters_NoSequenceID(t *testing.T) {
	parameters := make(map[string]*grpc_generated_v2.InferParameter)
	wrapper := NewRequestWrapper(
		"",
		"",
		nil,
		nil,
		nil,
	)
	wrapper.addSequenceParameters(parameters)
	if _, ok := parameters["sequence_id"]; ok {
		t.Errorf("Did not expect sequence_id parameter, got %v", parameters["sequence_id"])
	}
	if _, ok := parameters["sequence_start"]; ok {
		t.Errorf("Did not expect sequence_start parameter, got %v", parameters["sequence_start"])
	}
	if _, ok := parameters["sequence_end"]; ok {
		t.Errorf("Did not expect sequence_end parameter, got %v", parameters["sequence_end"])
	}
}

func TestAddPriorityAndTimeout(t *testing.T) {
	parameters := make(map[string]*grpc_generated_v2.InferParameter)
	priority := 1
	timeout := 1000
	wrapper := NewRequestWrapper(
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
	if param, ok := parameters["priority"]; !ok || param.GetUint64Param() != uint64(priority) {
		t.Errorf("Expected priority %d, got %v", priority, param)
	}
	if param, ok := parameters["timeout"]; !ok || param.GetInt64Param() != int64(timeout) {
		t.Errorf("Expected timeout %d, got %v", timeout, param)
	}
}

func TestAddPriorityAndTimeout_Nil(t *testing.T) {
	parameters := make(map[string]*grpc_generated_v2.InferParameter)
	wrapper := NewRequestWrapper(
		"",
		"",
		nil,
		nil,

		nil,
	)
	wrapper.addPriorityAndTimeout(parameters)
	if _, ok := parameters["priority"]; ok {
		t.Errorf("Did not expect priority parameter, got %v", parameters["priority"])
	}
	if _, ok := parameters["timeout"]; ok {
		t.Errorf("Did not expect timeout parameter, got %v", parameters["timeout"])
	}
}

func TestAddCustomParameters_SupportedTypes(t *testing.T) {
	parameters := make(map[string]*grpc_generated_v2.InferParameter)
	wrapper := NewRequestWrapper(
		"test_model",
		"",
		nil,
		nil,

		&options.InferOptions{
			Parameters: map[string]any{
				"string_param": "value",
				"bool_param":   true,
				"int_param":    int(42),
				"int64_param":  int64(64),
				"uint64_param": uint64(128),
				"float_param":  3.14,
			},
		},
	)
	err := wrapper.addCustomParameters(parameters)
	if err != nil {
		t.Fatalf("addCustomParameters returned error: %v", err)
	}
	if param, ok := parameters["string_param"]; !ok || param.GetStringParam() != "value" {
		t.Errorf("Expected string_param 'value', got %v", param)
	}
	if param, ok := parameters["bool_param"]; !ok || param.GetBoolParam() != true {
		t.Errorf("Expected bool_param true, got %v", param)
	}
	if param, ok := parameters["int_param"]; !ok || param.GetInt64Param() != 42 {
		t.Errorf("Expected int_param 42, got %v", param)
	}
	if param, ok := parameters["int64_param"]; !ok || param.GetInt64Param() != 64 {
		t.Errorf("Expected int64_param 64, got %v", param)
	}
	if param, ok := parameters["uint64_param"]; !ok || param.GetUint64Param() != 128 {
		t.Errorf("Expected uint64_param 128, got %v", param)
	}
	if param, ok := parameters["float_param"]; !ok || param.GetDoubleParam() != 3.14 {
		t.Errorf("Expected float_param 3.14, got %v", param)
	}
}

func TestAddCustomParameters_UnsupportedType(t *testing.T) {
	parameters := make(map[string]*grpc_generated_v2.InferParameter)
	wrapper := NewRequestWrapper(
		"test_model",
		"",
		nil,
		nil,
		&options.InferOptions{
			Parameters: map[string]any{
				"custom_param": []int{1, 2, 3},
			},
		},
	)
	err := wrapper.addCustomParameters(parameters)
	if err == nil {
		t.Fatal("Expected error for unsupported parameter type, got nil")
	}
}

func TestAddCustomParameters_ReservedKey(t *testing.T) {
	reservedKeys := []string{"sequence_id", "sequence_start", "sequence_end", "priority", "binary_data_output"}
	for _, key := range reservedKeys {
		parameters := make(map[string]*grpc_generated_v2.InferParameter)
		wrapper := NewRequestWrapper(
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

func TestConvertInputsToTensors_WithBinaryData(t *testing.T) {
	inputData := []int32{1, 2, 3, 4}
	input := NewInferInput("input0", "INT32", []int64{2, 2}, nil)
	input.SetData(inputData, true)
	wrapper := NewRequestWrapper(
		"",
		"",
		[]base.InferInput{input},
		nil,

		nil,
	)
	inputs, rawContents := wrapper.convertInputsToTensors()
	if len(inputs) != 1 {
		t.Errorf("Expected 1 input tensor, got %d", len(inputs))
	}
	if len(rawContents) != 1 {
		t.Errorf("Expected 1 raw content, got %d", len(rawContents))
	}
}

func TestConvertInputsToTensors_NoBinaryData(t *testing.T) {
	input := NewInferInput("input0", "INT32", []int64{2, 2}, nil)
	wrapper := NewRequestWrapper(
		"",
		"",
		[]base.InferInput{input},
		nil,
		nil,
	)
	inputs, rawContents := wrapper.convertInputsToTensors()
	if len(inputs) != 1 {
		t.Errorf("Expected 1 input tensor, got %d", len(inputs))
	}
	if len(rawContents) != 0 {
		t.Errorf("Expected 0 raw contents, got %d", len(rawContents))
	}
}

func TestConvertOutputsToTensors(t *testing.T) {
	output := NewInferOutput("output0", nil)
	wrapper := NewRequestWrapper(
		"",
		"",
		nil,
		[]base.InferOutput{output},
		nil,
	)
	outputs := wrapper.convertOutputsToTensors()
	if len(outputs) != 1 {
		t.Errorf("Expected 1 output tensor, got %d", len(outputs))
	}
}
