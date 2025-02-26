package http

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"errors"
	"github.com/Trendyol/go-triton-client/mocks"
	"go.uber.org/mock/gomock"
	"strconv"
	"testing"
)

func TestNewInferResult_NoHeaderLength(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2]}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)
	result, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_WithHeaderLength(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)

	result, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_UnmarshalError(t *testing.T) {
	body := []byte(`{"invalid_json"`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)

	_, err := NewInferResult(mockResponseWrapper, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestNewInferResult_Gzip(t *testing.T) {
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[]}`)
	gw.Write(body)
	gw.Close()
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)

	result, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_Deflate(t *testing.T) {
	var buf bytes.Buffer
	zw := zlib.NewWriter(&buf)
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[]}`)
	zw.Write(body)
	zw.Close()
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)

	result, err := NewInferResult(mockResponseWrapper, true)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
}

func TestNewInferResult_InvalidContentEncoding(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("invalid")
	mockResponseWrapper.EXPECT().GetBody().Return([]byte(`data`), nil)

	_, err := NewInferResult(mockResponseWrapper, true)
	if err == nil {
		t.Errorf("Expected error")
	}
}

func TestInferResult_AsFloat32Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, 16)...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsFloat32Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsFloat16Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP16","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, 16)...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsFloat16Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsFloat64Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP64","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, 16)...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsFloat64Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsInt8Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"INT8","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, int8(16))...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsInt8Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsInt16Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"INT16","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, int16(16))...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsInt16Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsInt32Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"INT32","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, int32(16))...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsInt32Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsInt64Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"INT64","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, int64(16))...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsInt64Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsUint8Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"UINT8","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, uint8(16))...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsUint8Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsUint16Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"UINT16","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, uint16(16))...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsUint16Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsUint32Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"UINT32","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, uint32(16))...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsUint32Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsUint64Slice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"UINT64","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, uint64(16))...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsUint64Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsBoolSlice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"BOOL","shape":[2,2],"parameters":{"binary_data_size":1}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, 1)...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsBoolSlice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsByteSlice_HasBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"BYTES","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, 16)...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsByteSlice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsFloat32Slice_NoBinaryData(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"data":[1.0,2.0,3.0,4.0]}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsFloat32Slice("output0")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if data == nil {
		t.Error("Expected data, got nil")
	}
}

func TestInferResult_AsFloat32Slice_NoBinaryData_ConvertError(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"data":[1.0,2.0,3.0,4.0]}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	data, err := result.AsFloat64Slice("output0")
	if err == nil {
		t.Errorf("Expected error, got nil")
	}
	if data != nil {
		t.Errorf("Expected nil, got data %v", data)
	}
}

func TestInferResult_GetOutput_NotFound(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output1","datatype":"FP32","shape":[2,2]}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(body, nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	_, err := result.GetOutput("output0")
	if err == nil || err.Error() != "output output0 not found" {
		t.Errorf("Expected error 'output output0 not found', got %v", err)
	}
}

func TestInferResult_AsSlice_NotFound(t *testing.T) {
	body := []byte(`{"model_name":"test_model","model_version":"1","outputs":[{"name":"output0","datatype":"FP32","shape":[2,2],"parameters":{"binary_data_size":16}}]}`)
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return(strconv.Itoa(len(body)))
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(append(body, make([]byte, 16)...), nil)

	result, _ := NewInferResult(mockResponseWrapper, true)
	_, err := result.AsFloat32Slice("output1")
	if err == nil || err.Error() != "output output1 not found" {
		t.Errorf("Expected error 'output output1 not found', got %v", err)
	}
}

func TestNewInferResult_BodyError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return(nil, errors.New("body error"))

	_, err := NewInferResult(mockResponseWrapper, true)
	if err == nil || err.Error() != "body error" {
		t.Errorf("Expected error 'body error', got %v", err)
	}
}

func TestNewInferResult_Deflate_ReadAllError(t *testing.T) {
	var buf bytes.Buffer
	zw := zlib.NewWriter(&buf)
	zw.Write([]byte(`{"model_name":"test_model","model_version":"1","outputs":[]}`))
	zw.Close()
	faultyData := buf.Bytes()[:10]
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("deflate")
	mockResponseWrapper.EXPECT().GetBody().Return(faultyData, nil)

	_, err := NewInferResult(mockResponseWrapper, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestNewInferResult_Gzip_ReadAllError(t *testing.T) {
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	gw.Write([]byte(`{"model_name":"test_model","model_version":"1","outputs":[]}`))
	gw.Close()
	faultyData := buf.Bytes()[:10]
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("gzip")
	mockResponseWrapper.EXPECT().GetBody().Return(faultyData, nil)

	_, err := NewInferResult(mockResponseWrapper, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestNewInferResult_GzipReaderError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("gzip")
	mockResponseWrapper.EXPECT().GetBody().Return([]byte("invalid gzip data"), nil)

	_, err := NewInferResult(mockResponseWrapper, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestNewInferResult_ZlibReaderError(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("deflate")
	mockResponseWrapper.EXPECT().GetBody().Return([]byte("invalid deflate data"), nil)

	_, err := NewInferResult(mockResponseWrapper, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestNewInferResult_InvalidHeaderLength(t *testing.T) {
	mockController := gomock.NewController(t)
	defer mockController.Finish()
	mockResponseWrapper := mocks.NewMockResponseWrapper(mockController)
	mockResponseWrapper.EXPECT().GetHeader("Inference-Header-Content-Length").Return("invalid")
	mockResponseWrapper.EXPECT().GetHeader("Content-Encoding").Return("")
	mockResponseWrapper.EXPECT().GetBody().Return([]byte("data"), nil)

	_, err := NewInferResult(mockResponseWrapper, true)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}
