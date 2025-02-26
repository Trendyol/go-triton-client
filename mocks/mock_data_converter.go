// Code generated by MockGen. DO NOT EDIT.
// Source: converter/data_converter.go
//
// Generated by this command:
//
//	mockgen -source=converter/data_converter.go -destination=mocks/mock_data_converter.go -package=mocks
//

// Package mocks is a generated GoMock package.
package mocks

import (
	reflect "reflect"

	gomock "go.uber.org/mock/gomock"
)

// MockDataConverter is a mock of DataConverter interface.
type MockDataConverter struct {
	ctrl     *gomock.Controller
	recorder *MockDataConverterMockRecorder
	isgomock struct{}
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

// ConvertInterfaceSliceToBoolSliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToBoolSliceAsInterface(data []any) []any {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToBoolSliceAsInterface", data)
	ret0, _ := ret[0].([]any)
	return ret0
}

// ConvertInterfaceSliceToBoolSliceAsInterface indicates an expected call of ConvertInterfaceSliceToBoolSliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToBoolSliceAsInterface(data any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToBoolSliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToBoolSliceAsInterface), data)
}

// ConvertInterfaceSliceToBytesSliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToBytesSliceAsInterface(data []any) ([]any, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToBytesSliceAsInterface", data)
	ret0, _ := ret[0].([]any)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ConvertInterfaceSliceToBytesSliceAsInterface indicates an expected call of ConvertInterfaceSliceToBytesSliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToBytesSliceAsInterface(data any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToBytesSliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToBytesSliceAsInterface), data)
}

// ConvertInterfaceSliceToFloat32SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToFloat32SliceAsInterface(data []any) []any {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToFloat32SliceAsInterface", data)
	ret0, _ := ret[0].([]any)
	return ret0
}

// ConvertInterfaceSliceToFloat32SliceAsInterface indicates an expected call of ConvertInterfaceSliceToFloat32SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToFloat32SliceAsInterface(data any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToFloat32SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToFloat32SliceAsInterface), data)
}

// ConvertInterfaceSliceToFloat64SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToFloat64SliceAsInterface(data []any) []any {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToFloat64SliceAsInterface", data)
	ret0, _ := ret[0].([]any)
	return ret0
}

// ConvertInterfaceSliceToFloat64SliceAsInterface indicates an expected call of ConvertInterfaceSliceToFloat64SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToFloat64SliceAsInterface(data any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToFloat64SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToFloat64SliceAsInterface), data)
}

// ConvertInterfaceSliceToInt32SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToInt32SliceAsInterface(data []any) []any {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToInt32SliceAsInterface", data)
	ret0, _ := ret[0].([]any)
	return ret0
}

// ConvertInterfaceSliceToInt32SliceAsInterface indicates an expected call of ConvertInterfaceSliceToInt32SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToInt32SliceAsInterface(data any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToInt32SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToInt32SliceAsInterface), data)
}

// ConvertInterfaceSliceToInt64SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToInt64SliceAsInterface(data []any) []any {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToInt64SliceAsInterface", data)
	ret0, _ := ret[0].([]any)
	return ret0
}

// ConvertInterfaceSliceToInt64SliceAsInterface indicates an expected call of ConvertInterfaceSliceToInt64SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToInt64SliceAsInterface(data any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToInt64SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToInt64SliceAsInterface), data)
}

// ConvertInterfaceSliceToUint32SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToUint32SliceAsInterface(data []any) []any {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToUint32SliceAsInterface", data)
	ret0, _ := ret[0].([]any)
	return ret0
}

// ConvertInterfaceSliceToUint32SliceAsInterface indicates an expected call of ConvertInterfaceSliceToUint32SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToUint32SliceAsInterface(data any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToUint32SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToUint32SliceAsInterface), data)
}

// ConvertInterfaceSliceToUint64SliceAsInterface mocks base method.
func (m *MockDataConverter) ConvertInterfaceSliceToUint64SliceAsInterface(data []any) []any {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ConvertInterfaceSliceToUint64SliceAsInterface", data)
	ret0, _ := ret[0].([]any)
	return ret0
}

// ConvertInterfaceSliceToUint64SliceAsInterface indicates an expected call of ConvertInterfaceSliceToUint64SliceAsInterface.
func (mr *MockDataConverterMockRecorder) ConvertInterfaceSliceToUint64SliceAsInterface(data any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ConvertInterfaceSliceToUint64SliceAsInterface", reflect.TypeOf((*MockDataConverter)(nil).ConvertInterfaceSliceToUint64SliceAsInterface), data)
}

// DeserializeBF16Tensor mocks base method.
func (m *MockDataConverter) DeserializeBF16Tensor(encodedTensor []byte) ([]float32, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeBF16Tensor", encodedTensor)
	ret0, _ := ret[0].([]float32)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeBF16Tensor indicates an expected call of DeserializeBF16Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeBF16Tensor(encodedTensor any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeBF16Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeBF16Tensor), encodedTensor)
}

// DeserializeBoolTensor mocks base method.
func (m *MockDataConverter) DeserializeBoolTensor(dataBuffer []byte) ([]bool, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeBoolTensor", dataBuffer)
	ret0, _ := ret[0].([]bool)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeBoolTensor indicates an expected call of DeserializeBoolTensor.
func (mr *MockDataConverterMockRecorder) DeserializeBoolTensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeBoolTensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeBoolTensor), dataBuffer)
}

// DeserializeBytesTensor mocks base method.
func (m *MockDataConverter) DeserializeBytesTensor(encodedTensor []byte) ([]string, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeBytesTensor", encodedTensor)
	ret0, _ := ret[0].([]string)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeBytesTensor indicates an expected call of DeserializeBytesTensor.
func (mr *MockDataConverterMockRecorder) DeserializeBytesTensor(encodedTensor any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeBytesTensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeBytesTensor), encodedTensor)
}

// DeserializeFloat16Tensor mocks base method.
func (m *MockDataConverter) DeserializeFloat16Tensor(dataBuffer []byte) ([]float64, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeFloat16Tensor", dataBuffer)
	ret0, _ := ret[0].([]float64)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeFloat16Tensor indicates an expected call of DeserializeFloat16Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeFloat16Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeFloat16Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeFloat16Tensor), dataBuffer)
}

// DeserializeFloat32Tensor mocks base method.
func (m *MockDataConverter) DeserializeFloat32Tensor(dataBuffer []byte) ([]float32, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeFloat32Tensor", dataBuffer)
	ret0, _ := ret[0].([]float32)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeFloat32Tensor indicates an expected call of DeserializeFloat32Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeFloat32Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeFloat32Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeFloat32Tensor), dataBuffer)
}

// DeserializeFloat64Tensor mocks base method.
func (m *MockDataConverter) DeserializeFloat64Tensor(dataBuffer []byte) ([]float64, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeFloat64Tensor", dataBuffer)
	ret0, _ := ret[0].([]float64)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeFloat64Tensor indicates an expected call of DeserializeFloat64Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeFloat64Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeFloat64Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeFloat64Tensor), dataBuffer)
}

// DeserializeInt16Tensor mocks base method.
func (m *MockDataConverter) DeserializeInt16Tensor(dataBuffer []byte) ([]int16, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeInt16Tensor", dataBuffer)
	ret0, _ := ret[0].([]int16)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeInt16Tensor indicates an expected call of DeserializeInt16Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeInt16Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeInt16Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeInt16Tensor), dataBuffer)
}

// DeserializeInt32Tensor mocks base method.
func (m *MockDataConverter) DeserializeInt32Tensor(dataBuffer []byte) ([]int32, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeInt32Tensor", dataBuffer)
	ret0, _ := ret[0].([]int32)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeInt32Tensor indicates an expected call of DeserializeInt32Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeInt32Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeInt32Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeInt32Tensor), dataBuffer)
}

// DeserializeInt64Tensor mocks base method.
func (m *MockDataConverter) DeserializeInt64Tensor(dataBuffer []byte) ([]int64, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeInt64Tensor", dataBuffer)
	ret0, _ := ret[0].([]int64)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeInt64Tensor indicates an expected call of DeserializeInt64Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeInt64Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeInt64Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeInt64Tensor), dataBuffer)
}

// DeserializeInt8Tensor mocks base method.
func (m *MockDataConverter) DeserializeInt8Tensor(dataBuffer []byte) ([]int8, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeInt8Tensor", dataBuffer)
	ret0, _ := ret[0].([]int8)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeInt8Tensor indicates an expected call of DeserializeInt8Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeInt8Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeInt8Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeInt8Tensor), dataBuffer)
}

// DeserializeUint16Tensor mocks base method.
func (m *MockDataConverter) DeserializeUint16Tensor(dataBuffer []byte) ([]uint16, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeUint16Tensor", dataBuffer)
	ret0, _ := ret[0].([]uint16)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeUint16Tensor indicates an expected call of DeserializeUint16Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeUint16Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeUint16Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeUint16Tensor), dataBuffer)
}

// DeserializeUint32Tensor mocks base method.
func (m *MockDataConverter) DeserializeUint32Tensor(dataBuffer []byte) ([]uint32, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeUint32Tensor", dataBuffer)
	ret0, _ := ret[0].([]uint32)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeUint32Tensor indicates an expected call of DeserializeUint32Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeUint32Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeUint32Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeUint32Tensor), dataBuffer)
}

// DeserializeUint64Tensor mocks base method.
func (m *MockDataConverter) DeserializeUint64Tensor(dataBuffer []byte) ([]uint64, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeUint64Tensor", dataBuffer)
	ret0, _ := ret[0].([]uint64)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeUint64Tensor indicates an expected call of DeserializeUint64Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeUint64Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeUint64Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeUint64Tensor), dataBuffer)
}

// DeserializeUint8Tensor mocks base method.
func (m *MockDataConverter) DeserializeUint8Tensor(dataBuffer []byte) ([]uint8, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeserializeUint8Tensor", dataBuffer)
	ret0, _ := ret[0].([]uint8)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeserializeUint8Tensor indicates an expected call of DeserializeUint8Tensor.
func (mr *MockDataConverterMockRecorder) DeserializeUint8Tensor(dataBuffer any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeserializeUint8Tensor", reflect.TypeOf((*MockDataConverter)(nil).DeserializeUint8Tensor), dataBuffer)
}

// FlattenData mocks base method.
func (m *MockDataConverter) FlattenData(inputTensor any) []any {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "FlattenData", inputTensor)
	ret0, _ := ret[0].([]any)
	return ret0
}

// FlattenData indicates an expected call of FlattenData.
func (mr *MockDataConverterMockRecorder) FlattenData(inputTensor any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "FlattenData", reflect.TypeOf((*MockDataConverter)(nil).FlattenData), inputTensor)
}

// SerializeTensor mocks base method.
func (m *MockDataConverter) SerializeTensor(inputTensor any) ([]byte, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "SerializeTensor", inputTensor)
	ret0, _ := ret[0].([]byte)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// SerializeTensor indicates an expected call of SerializeTensor.
func (mr *MockDataConverterMockRecorder) SerializeTensor(inputTensor any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SerializeTensor", reflect.TypeOf((*MockDataConverter)(nil).SerializeTensor), inputTensor)
}
