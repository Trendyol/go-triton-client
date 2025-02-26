package grpc

import (
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"github.com/Trendyol/go-triton-client/mocks"
	"testing"

	"go.uber.org/mock/gomock"
)

func TestNewInferResult_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	result, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_InvalidResponseType(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	mockResponseWrapper.EXPECT().GetResponse().Return("invalid_response").Times(1)

	result, err := NewInferResult(mockResponseWrapper, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if err.Error() != "invalid response type" {
		t.Errorf("Expected error 'invalid response type', got %v", err)
	}
	if result != nil {
		t.Error("Expected result to be nil")
	}
}

func TestNewInferResult_Verbose(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	result, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_NoOutputs(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:         "test_model",
		ModelVersion:      "1",
		Outputs:           []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{},
		RawOutputContents: [][]byte{},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	result, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestInferResult_AsFloat16Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP16",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsFloat16Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsFloat32Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsFloat32Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsFloat64Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP64",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsFloat64Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsInt8Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "INT8",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, int8(16))},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsInt8Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsInt16Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "INT16",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, int16(16))},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsInt16Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsInt32Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "INT32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, int32(16))},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsInt32Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsInt64Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "INT64",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, int64(16))},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsInt64Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsUint8Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "UINT8",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, uint8(16))},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsUint8Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsUint16Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "UINT16",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, uint16(16))},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsUint16Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsUint32Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "UINT32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, uint32(16))},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsUint32Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsUint64Slice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "UINT64",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, uint64(16))},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsUint64Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsBoolSlice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "BOOL",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 1)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsBoolSlice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsBytesSlice_Success(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "BYTES",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	data, err := result.AsByteSlice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsSlice_OutputNotFound(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:       "output1",
				Datatype:   "FP32",
				Shape:      []int64{1, 1},
				Parameters: nil,
				Contents:   nil,
			},
		},
		RawOutputContents: [][]byte{[]byte("test")},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	_, err = result.AsFloat32Slice("output0")
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if err.Error() != "output output0 not found" {
		t.Errorf("Expected error 'output output0 not found', got %v", err)
	}
}

func TestInferResult_GetOutput_Found(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output0",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	output, err := result.GetOutput("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if output.GetName() != "output0" {
		t.Errorf("Expected output name 'output0', got '%s'", output.GetName())
	}
}

func TestInferResult_GetOutput_NotFound(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:       "output1",
				Datatype:   "FP32",
				Shape:      []int64{1, 1},
				Parameters: nil,
				Contents:   nil,
			},
		},
		RawOutputContents: [][]byte{[]byte("test")},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	_, err = result.GetOutput("output0")
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if err.Error() != "output output0 not found" {
		t.Errorf("Expected error 'output output0 not found', got %v", err)
	}
}

func TestInferResult_AsSlice_MissingBufferMapEntry(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()

	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)

	response := &grpc_generated_v2.ModelInferResponse{
		ModelName:    "test_model",
		ModelVersion: "1",
		Outputs: []*grpc_generated_v2.ModelInferResponse_InferOutputTensor{
			{
				Name:     "output1",
				Datatype: "FP32",
				Shape:    []int64{2, 2},
			},
		},
		RawOutputContents: [][]byte{make([]byte, 16)},
	}

	mockResponseWrapper.EXPECT().GetResponse().Return(response).Times(1)

	resultInterface, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	result := resultInterface.(*InferResult)

	_, err = result.AsFloat32Slice("output0")
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if err.Error() != "output output0 not found" {
		t.Errorf("Expected error 'output output0 not found', got %v", err)
	}
}
