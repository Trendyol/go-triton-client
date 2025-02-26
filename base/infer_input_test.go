package base

import (
	"github.com/stretchr/testify/assert"
	"reflect"
	"testing"
)

func TestBaseInferInput_GetName(t *testing.T) {
	input := &BaseInferInput{Name: "input0"}
	if input.GetName() != "input0" {
		t.Errorf("Expected Name 'input0', got %s", input.GetName())
	}
}

func TestBaseInferInput_GetShape(t *testing.T) {
	shape := []int64{1, 3}
	input := &BaseInferInput{Shape: shape}
	if !reflect.DeepEqual(input.GetShape(), shape) {
		t.Errorf("Expected Shape %v, got %v", shape, input.GetShape())
	}
}

func TestBaseInferInput_GetDatatype(t *testing.T) {
	input := &BaseInferInput{Datatype: "FP32"}
	if input.GetDatatype() != "FP32" {
		t.Errorf("Expected Datatype 'FP32', got %s", input.GetDatatype())
	}
}

func TestBaseInferInput_GetParameters(t *testing.T) {
	parameters := map[string]any{"param1": "value1"}
	input := &BaseInferInput{Parameters: parameters}
	if !reflect.DeepEqual(input.GetParameters(), parameters) {
		t.Errorf("Expected Parameters %v, got %v", parameters, input.GetParameters())
	}
}

func TestBaseInferInput_GetData(t *testing.T) {
	data := []any{1, 2, 3}
	input := &BaseInferInput{Data: data}
	if !reflect.DeepEqual(input.GetData(), data) {
		t.Errorf("Expected Data %v, got %v", data, input.GetData())
	}
}

func TestBaseInferInput_GetRawData(t *testing.T) {
	rawData := []byte{1, 2, 3, 4}
	input := &BaseInferInput{RawData: rawData}
	if !reflect.DeepEqual(input.GetRawData(), rawData) {
		t.Errorf("Expected RawData %v, got %v", rawData, input.GetRawData())
	}
}

func TestBaseInferInput_SetData(t *testing.T) {
	inputTensor := []int32{1, 2, 3}
	datatype := "INT32"
	input := &BaseInferInput{
		Datatype:   datatype,
		Parameters: map[string]any{},
	}
	err := input.SetData(inputTensor, true)
	if err != nil {
		t.Fatalf("SetData returned error: %v", err)
	}
	size, ok := input.Parameters["binary_data_size"]
	if !ok {
		t.Error("Expected 'binary_data_size' parameter to be not deleted")
	}
	assert.Equal(t, 3*4, size)
	if input.Data != nil {
		t.Errorf("Expected RawData to be nil, got %v", input.RawData)
	}
	expectedData := []byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0}
	if !reflect.DeepEqual(input.RawData, expectedData) {
		t.Errorf("Expected Data %v, got %v", expectedData, input.Data)
	}
}

func TestBaseInferInput_SetData_NonBinaryData(t *testing.T) {
	inputTensor := []int32{1, 2, 3}
	datatype := "INT32"
	input := &BaseInferInput{
		Datatype:   datatype,
		Parameters: map[string]any{"binary_data_size": 100},
	}
	err := input.SetData(inputTensor, false)
	if err != nil {
		t.Fatalf("SetData returned error: %v", err)
	}
	if _, ok := input.Parameters["binary_data_size"]; ok {
		t.Error("Expected 'binary_data_size' parameter to be deleted")
	}
	if input.RawData != nil {
		t.Errorf("Expected RawData to be nil, got %v", input.RawData)
	}
	expectedData := []any{int32(1), int32(2), int32(3)}
	if !reflect.DeepEqual(input.Data, expectedData) {
		t.Errorf("Expected Data %v, got %v", expectedData, input.Data)
	}
}

func TestBaseInferInput_SetData_InvalidDatatype(t *testing.T) {
	inputTensor := []float32{1.0, 2.0, 3.0}
	datatype := "INT32"
	input := &BaseInferInput{
		Datatype:   datatype,
		Parameters: make(map[string]any),
	}
	err := input.SetData(inputTensor, false)
	if err == nil {
		t.Fatal("Expected error due to mismatched datatype, got nil")
	}
}

func TestGetDatatype(t *testing.T) {
	tests := []struct {
		input    any
		expected string
	}{
		{[]int8{1, 2}, "INT8"},
		{[]int16{int16(1), int16(1)}, "INT16"},
		{[]int32{1, 2}, "INT32"},
		{[]int64{1, 2}, "INT64"},
		{[]uint16{1, 2}, "UINT16"},
		{[]uint32{1, 2}, "UINT32"},
		{[]uint64{1, 2}, "UINT64"},
		{[]float32{1.0, 2.0}, "FP32"},
		{[]float64{1.0, 2.0}, "FP64"},
		{[]byte{1, 2}, "BYTES"},
		{[]bool{true, false}, "BOOL"},
		{[]string{"a", "b"}, "BYTES"},
		{[]any{1 + 2i}, "UNKNOWN"},
	}
	for _, tt := range tests {
		datatype := GetDatatype(tt.input)
		if datatype != tt.expected {
			t.Errorf("getDatatype(%T) = %s, expected %s", tt.input, datatype, tt.expected)
		}
	}
}

func TestBaseInferInput_SetShape(t *testing.T) {
	shape := []int64{1, 3}
	input := &BaseInferInput{}
	input.SetShape(shape)
	if !reflect.DeepEqual(input.GetShape(), shape) {
		t.Errorf("Expected Shape %v, got %v", shape, input.GetShape())
	}
}

func TestBaseInferInput_SetDatatype(t *testing.T) {
	input := &BaseInferInput{}
	input.SetDatatype("FP32")
	if input.GetDatatype() != "FP32" {
		t.Errorf("Expected Datatype 'FP32', got %s", input.GetDatatype())
	}
}
