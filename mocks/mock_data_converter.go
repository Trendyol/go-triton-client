// Code generated by MockGen. DO NOT EDIT.
// Source: converter/data_converter.go

// Package mocks is a generated GoMock package.
package mocks

import (
	reflect "reflect"

	gomock "github.com/golang/mock/gomock"
)

// MockDataConverter is a mock of DataConverter interface.
type MockDataConverter struct {
	ctrl     *gomock.Controller
	recorder *MockDataConverterMockRecorder
}

// MockDataConverterMockRecorder is the mock recorder for MockDataConverter.
type MockDataConverterMockRecorder struct {
	mock *MockDataConverter
}

// NewMockDataConverter creates a new mock instance.
func NewMockDataConverter(ctrl *gomock.Controller) *MockDataConverter {
	mock := &MockDataConverter{ctrl: ctrl}
	mock.recorder = &MockDataConverterMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockDataConverter) EXPECT() *MockDataConverterMockRecorder {
	return m.recorder
}

// ConvertByteSliceToFloat32Slice mocks base method.
func (m *MockDataConverter) ConvertByteSliceToFloat32Slice(data []byte) []float32 {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertByteSliceToFloat32Slice", data)
	ret0, _ := ret[0].([]float32)
	return ret0
}

// ConvertByteSliceToFloat32Slice indicates an expected call of ConvertByteSliceToFloat32Slice.
func (mr *MockDataConverterMockRecorder) ConvertByteSliceToFloat32Slice(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertByteSliceToFloat32Slice", reflect.TypeOf((*MockDataConverter)(nil).ConvertByteSliceToFloat32Slice), data)
}

// ConvertByteSliceToFloat64Slice mocks base method.
func (m *MockDataConverter) ConvertByteSliceToFloat64Slice(data []byte) []float64 {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertByteSliceToFloat64Slice", data)
	ret0, _ := ret[0].([]float64)
	return ret0
}

// ConvertByteSliceToFloat64Slice indicates an expected call of ConvertByteSliceToFloat64Slice.
func (mr *MockDataConverterMockRecorder) ConvertByteSliceToFloat64Slice(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertByteSliceToFloat64Slice", reflect.TypeOf((*MockDataConverter)(nil).ConvertByteSliceToFloat64Slice), data)
}

// ConvertByteSliceToInt32Slice mocks base method.
func (m *MockDataConverter) ConvertByteSliceToInt32Slice(data []byte) []int32 {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertByteSliceToInt32Slice", data)
	ret0, _ := ret[0].([]int32)
	return ret0
}

// ConvertByteSliceToInt32Slice indicates an expected call of ConvertByteSliceToInt32Slice.
func (mr *MockDataConverterMockRecorder) ConvertByteSliceToInt32Slice(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertByteSliceToInt32Slice", reflect.TypeOf((*MockDataConverter)(nil).ConvertByteSliceToInt32Slice), data)
}

// ConvertByteSliceToInt64Slice mocks base method.
func (m *MockDataConverter) ConvertByteSliceToInt64Slice(data []byte) []int64 {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertByteSliceToInt64Slice", data)
	ret0, _ := ret[0].([]int64)
	return ret0
}

// ConvertByteSliceToInt64Slice indicates an expected call of ConvertByteSliceToInt64Slice.
func (mr *MockDataConverterMockRecorder) ConvertByteSliceToInt64Slice(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertByteSliceToInt64Slice", reflect.TypeOf((*MockDataConverter)(nil).ConvertByteSliceToInt64Slice), data)
}

// ConvertInterfaceSliceToBoolSliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToBoolSliceAsInterface(data []interface{}) []interface{} {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToBoolSliceAsInterface", data)
	ret0, _ := ret[0].([]interface{})
	return ret0
}

// ConvertInterfaceSliceToBoolSliceAsInterface indicates an expected call of ConvertInterfaceSliceToBoolSliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToBoolSliceAsInterface(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToBoolSliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToBoolSliceAsInterface), data)
}

// ConvertInterfaceSliceToBytesSliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToBytesSliceAsInterface(data []interface{}) ([]interface{}, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToBytesSliceAsInterface", data)
	ret0, _ := ret[0].([]interface{})
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ConvertInterfaceSliceToBytesSliceAsInterface indicates an expected call of ConvertInterfaceSliceToBytesSliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToBytesSliceAsInterface(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToBytesSliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToBytesSliceAsInterface), data)
}

// ConvertInterfaceSliceToFloat32SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToFloat32SliceAsInterface(data []interface{}) []interface{} {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToFloat32SliceAsInterface", data)
	ret0, _ := ret[0].([]interface{})
	return ret0
}

// ConvertInterfaceSliceToFloat32SliceAsInterface indicates an expected call of ConvertInterfaceSliceToFloat32SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToFloat32SliceAsInterface(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToFloat32SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToFloat32SliceAsInterface), data)
}

// ConvertInterfaceSliceToFloat64SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToFloat64SliceAsInterface(data []interface{}) []interface{} {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToFloat64SliceAsInterface", data)
	ret0, _ := ret[0].([]interface{})
	return ret0
}

// ConvertInterfaceSliceToFloat64SliceAsInterface indicates an expected call of ConvertInterfaceSliceToFloat64SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToFloat64SliceAsInterface(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToFloat64SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToFloat64SliceAsInterface), data)
}

// ConvertInterfaceSliceToInt32SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToInt32SliceAsInterface(data []interface{}) []interface{} {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToInt32SliceAsInterface", data)
	ret0, _ := ret[0].([]interface{})
	return ret0
}

// ConvertInterfaceSliceToInt32SliceAsInterface indicates an expected call of ConvertInterfaceSliceToInt32SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToInt32SliceAsInterface(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToInt32SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToInt32SliceAsInterface), data)
}

// ConvertInterfaceSliceToInt64SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToInt64SliceAsInterface(data []interface{}) []interface{} {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToInt64SliceAsInterface", data)
	ret0, _ := ret[0].([]interface{})
	return ret0
}

// ConvertInterfaceSliceToInt64SliceAsInterface indicates an expected call of ConvertInterfaceSliceToInt64SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToInt64SliceAsInterface(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToInt64SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToInt64SliceAsInterface), data)
}

// ConvertInterfaceSliceToUint32SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToUint32SliceAsInterface(data []interface{}) []interface{} {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToUint32SliceAsInterface", data)
	ret0, _ := ret[0].([]interface{})
	return ret0
}

// ConvertInterfaceSliceToUint32SliceAsInterface indicates an expected call of ConvertInterfaceSliceToUint32SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToUint32SliceAsInterface(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToUint32SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToUint32SliceAsInterface), data)
}

// ConvertInterfaceSliceToUint64SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToUint64SliceAsInterface(data []interface{}) []interface{} {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToUint64SliceAsInterface", data)
	ret0, _ := ret[0].([]interface{})
	return ret0
}

// ConvertInterfaceSliceToUint64SliceAsInterface indicates an expected call of ConvertInterfaceSliceToUint64SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToUint64SliceAsInterface(data interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToUint64SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToUint64SliceAsInterface), data)
}

// DeserializeTensor mocks base method.
func (m *MockDataConverter) DeserializeTensor(datatype string, dataBuffer []byte) (interface{}, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeTensor", datatype, dataBuffer)
	ret0, _ := ret[0].(interface{})
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeTensor indicates an expected call of DeserializeTensor.
func (mr *MockDataConverterMockRecorder) DeserializeTensor(datatype, dataBuffer interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeTensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeTensor), datatype, dataBuffer)
}

// FlattenData mocks base method.
func (m *MockDataConverter) FlattenData(inputTensor interface{}) []interface{} {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "FlattenData", inputTensor)
	ret0, _ := ret[0].([]interface{})
	return ret0
}

// FlattenData indicates an expected call of FlattenData.
func (mr *MockDataConverterMockRecorder) FlattenData(inputTensor interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "FlattenData", reflect.TypeOf((*MockDataConverter)(nil).FlattenData), inputTensor)
}

// ReshapeArray mocks base method.
func (m *MockDataConverter) ReshapeArray(data interface{}, shape []int) ([]interface{}, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ReshapeArray", data, shape)
	ret0, _ := ret[0].([]interface{})
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ReshapeArray indicates an expected call of ReshapeArray.
func (mr *MockDataConverterMockRecorder) ReshapeArray(data, shape interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ReshapeArray", reflect.TypeOf((*MockDataConverter)(nil).ReshapeArray), data, shape)
}

// SerializeTensor mocks base method.
func (m *MockDataConverter) SerializeTensor(inputTensor interface{}) ([]byte, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "SerializeTensor", inputTensor)
	ret0, _ := ret[0].([]byte)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// SerializeTensor indicates an expected call of SerializeTensor.
func (mr *MockDataConverterMockRecorder) SerializeTensor(inputTensor interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SerializeTensor", reflect.TypeOf((*MockDataConverter)(nil).SerializeTensor), inputTensor)
}