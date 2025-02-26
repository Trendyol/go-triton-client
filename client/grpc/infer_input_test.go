package grpc

import (
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/client/grpc/grpc_generated_v2"
	"go.uber.org/mock/gomock"
	"reflect"
	"testing"
)

func TestNewInferInput(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	parameters := map[string]any{"param1": "value1"}
	input := NewInferInput(name, datatype, shape, parameters)
	if input.GetName() != name {
		t.Errorf("Expected Name %s, got %s", name, input.GetName())
	}
	if !reflect.DeepEqual(input.GetShape(), shape) {
		t.Errorf("Expected Shape %v, got %v", shape, input.GetShape())
	}
	if input.GetDatatype() != datatype {
		t.Errorf("Expected Datatype %s, got %s", datatype, input.GetDatatype())
	}
	if !reflect.DeepEqual(input.GetParameters(), parameters) {
		t.Errorf("Expected Parameters %v, got %v", parameters, input.GetParameters())
	}
}

func TestNewInferInput_NilParameters(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	input := NewInferInput(name, datatype, shape, nil)
	if input.GetParameters() == nil {
		t.Errorf("Expected Parameters to be initialized, got nil")
	}
}

func TestInferInput_GetTensor_WithRawData(t *testing.T) {
	name := "input0"
	datatype := "FP32"
	shape := []int64{1, 3}
	rawData := []byte{1, 2, 3, 4}
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			Name:     name,
			Datatype: datatype,
			Shape:    shape,
			RawData:  rawData,
		},
	}
	tensor := input.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferInputTensor)
	if tensor.Name != name {
		t.Errorf("Expected Name %s, got %s", name, tensor.Name)
	}
	if tensor.Datatype != datatype {
		t.Errorf("Expected Datatype %s, got %s", datatype, tensor.Datatype)
	}
	if !reflect.DeepEqual(tensor.Shape, shape) {
		t.Errorf("Expected Shape %v, got %v", shape, tensor.Shape)
	}
	if tensor.Contents != nil {
		t.Errorf("Expected Contents to be nil when RawData is present")
	}
}

func TestInferInput_GetTensor(t *testing.T) {
	testCases := []struct {
		name             string
		datatype         string
		shape            []int64
		rawData          []byte
		data             []any
		expectedContents *grpc_generated_v2.InferTensorContents
	}{
		{
			name:             "RawDataPresent",
			datatype:         "FP64",
			shape:            []int64{1, 1},
			rawData:          []byte("non-empty"),
			data:             nil,
			expectedContents: nil,
		},
		{
			name:     "FP64",
			datatype: "FP64",
			shape:    []int64{1, 3},
			rawData:  nil,
			data:     []any{float64(1.1), float64(2.2), float64(3.3)},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				Fp64Contents: []float64{1.1, 2.2, 3.3},
			},
		},
		{
			name:     "INT64",
			datatype: "INT64",
			shape:    []int64{1, 3},
			rawData:  nil,
			data:     []any{int64(10), int64(20), int64(30)},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				Int64Contents: []int64{10, 20, 30},
			},
		},
		{
			name:     "UINT8",
			datatype: "UINT8",
			shape:    []int64{1, 3},
			rawData:  nil,
			data:     []any{uint8(5), uint8(6), uint8(7)},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				UintContents: uInt8SliceToUint32Slice([]uint8{5, 6, 7}),
			},
		},
		{
			name:     "UINT16",
			datatype: "UINT16",
			shape:    []int64{1, 2},
			rawData:  nil,
			data:     []any{uint16(100), uint16(200)},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				UintContents: uInt16SliceToUint32Slice([]uint16{100, 200}),
			},
		},
		{
			name:     "UINT32",
			datatype: "UINT32",
			shape:    []int64{1, 2},
			rawData:  nil,
			data:     []any{uint32(300), uint32(400)},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				UintContents: []uint32{300, 400},
			},
		},
		{
			name:     "UINT64",
			datatype: "UINT64",
			shape:    []int64{1, 2},
			rawData:  nil,
			data:     []any{uint64(500), uint64(600)},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				Uint64Contents: []uint64{500, 600},
			},
		},
		{
			name:     "INT8",
			datatype: "INT8",
			shape:    []int64{1, 3},
			rawData:  nil,
			data:     []any{int8(-1), int8(0), int8(1)},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				IntContents: int8SliceToInt32Slice([]int8{-1, 0, 1}),
			},
		},
		{
			name:     "INT16",
			datatype: "INT16",
			shape:    []int64{1, 2},
			rawData:  nil,
			data:     []any{int16(-50), int16(50)},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				IntContents: int16SliceToInt32Slice([]int16{-50, 50}),
			},
		},
		{
			name:     "INT32",
			datatype: "INT32",
			shape:    []int64{1, 2},
			rawData:  nil,
			data:     []any{int32(1000), int32(2000)},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				IntContents: []int32{1000, 2000},
			},
		},
		{
			name:     "FP32",
			datatype: "FP32",
			shape:    []int64{1, 2},
			rawData:  nil,
			data:     []any{float32(1.5), float32(2.5)},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				Fp32Contents: []float32{1.5, 2.5},
			},
		},
		{
			name:     "BOOL",
			datatype: "BOOL",
			shape:    []int64{1, 2},
			rawData:  nil,
			data:     []any{true, false},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				BoolContents: []bool{true, false},
			},
		},
		{
			name:     "BYTES",
			datatype: "BYTES",
			shape:    []int64{1, 3},
			rawData:  nil,
			data:     []any{"foo"},
			expectedContents: &grpc_generated_v2.InferTensorContents{
				BytesContents: [][]byte{[]byte("foo")},
			},
		},
		{
			name:             "Unsupported Datatype",
			datatype:         "UNKNOWN",
			shape:            []int64{1, 1},
			rawData:          nil,
			data:             []any{"a", 1},
			expectedContents: nil,
		},
		{
			name:             "Deserialization Error",
			datatype:         "INT32",
			shape:            []int64{1, 1},
			rawData:          nil,
			data:             []any{float32(1.5)},
			expectedContents: nil,
		},
		{
			name:             "SerializeTensor Error",
			datatype:         "INT32",
			shape:            []int64{1, 1},
			rawData:          nil,
			data:             []any{float32(1.5)},
			expectedContents: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			input := &InferInput{
				BaseInferInput: &base.BaseInferInput{
					Name:     "input0",
					Datatype: tc.datatype,
					Shape:    tc.shape,
					RawData:  tc.rawData,
					Data:     tc.data,
				},
			}

			tensor := input.GetTensor().(*grpc_generated_v2.ModelInferRequest_InferInputTensor)

			if tensor.Name != "input0" {
				t.Errorf("Expected Name 'input0', got %s", tensor.Name)
			}
			if tensor.Datatype != tc.datatype {
				t.Errorf("Expected Datatype '%s', got '%s'", tc.datatype, tensor.Datatype)
			}
			if !reflect.DeepEqual(tensor.Shape, tc.shape) {
				t.Errorf("Expected Shape %v, got %v", tc.shape, tensor.Shape)
			}
			if !reflect.DeepEqual(tensor.Contents, tc.expectedContents) {
				t.Errorf("Expected Contents %+v, got %+v", tc.expectedContents, tensor.Contents)
			}
		})
	}
}

func TestInferInput_GetBinaryData(t *testing.T) {
	rawData := []byte{1, 2, 3, 4}
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{
			RawData: rawData,
		},
	}
	if !reflect.DeepEqual(input.GetBinaryData(), rawData) {
		t.Errorf("Expected GetBinaryData %v, got %v", rawData, input.GetBinaryData())
	}
}

func TestInferInput_GetBinaryData_Nil(t *testing.T) {
	input := &InferInput{
		BaseInferInput: &base.BaseInferInput{},
	}
	if input.GetBinaryData() != nil {
		t.Errorf("Expected GetBinaryData to be nil, got %v", input.GetBinaryData())
	}
}
