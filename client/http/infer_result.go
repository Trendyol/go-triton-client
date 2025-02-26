package http

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"encoding/json"
	"fmt"
	"github.com/Trendyol/go-triton-client/base"
	"github.com/Trendyol/go-triton-client/converter"
	"io"
	"strconv"
)

// InferResult represents the result of an inference operation using HTTP.
type InferResult struct {
	*base.BaseInferResult
}

// NewInferResult creates a new HTTP InferResult instance.
func NewInferResult(response base.ResponseWrapper, verbose bool) (base.InferResult, error) {
	headerLength := response.GetHeader("Inference-Header-Content-Length")

	var decompressedData []byte
	var err error

	contentEncoding := response.GetHeader("Content-Encoding")
	body, err := response.GetBody()
	if err != nil {
		return nil, err
	}

	switch contentEncoding {
	case "gzip":
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		decompressedData, err = io.ReadAll(reader)
		if err != nil {
			return nil, err
		}
	case "deflate":
		reader, err := zlib.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		decompressedData, err = io.ReadAll(reader)
		if err != nil {
			return nil, err
		}
	default:
		decompressedData = body
	}

	if headerLength == "" {
		content := decompressedData
		if verbose {
			fmt.Println(string(content))
		}
		var result base.InferOutputs
		if err := json.Unmarshal(content, &result); err != nil {
			return nil, err
		}

		return &InferResult{
			BaseInferResult: &base.BaseInferResult{
				OutputsResponse: result,
			},
		}, nil
	}

	headerLengthInt, err := strconv.Atoi(headerLength)
	if err != nil {
		return nil, err
	}
	content := decompressedData[:headerLengthInt]
	if verbose {
		fmt.Println(string(content))
	}

	var result base.InferOutputs
	if err := json.Unmarshal(content, &result); err != nil {
		return nil, err
	}

	buffer := decompressedData[headerLengthInt:]
	outputNameToBufferMap := make(map[string]int)
	bufferIndex := 0
	for _, output := range result.Outputs {
		thisDataSize, ok := output.GetParameters()["binary_data_size"].(float64)
		if ok {
			outputNameToBufferMap[output.GetName()] = bufferIndex
			bufferIndex += int(thisDataSize)
		}
	}

	return &InferResult{
		BaseInferResult: &base.BaseInferResult{
			OutputsResponse:       result,
			OutputNameToBufferMap: outputNameToBufferMap,
			Buffer:                buffer,
		},
	}, nil
}

func (r *InferResult) AsInt8Slice(name string) ([]int8, error) {
	return getAsSlice[int8](name, r, converter.DeserializeInt8Tensor)
}

func (r *InferResult) AsInt16Slice(name string) ([]int16, error) {
	return getAsSlice[int16](name, r, converter.DeserializeInt16Tensor)
}

func (r *InferResult) AsInt32Slice(name string) ([]int32, error) {
	return getAsSlice[int32](name, r, converter.DeserializeInt32Tensor)
}

func (r *InferResult) AsInt64Slice(name string) ([]int64, error) {
	return getAsSlice[int64](name, r, converter.DeserializeInt64Tensor)
}

func (r *InferResult) AsUint8Slice(name string) ([]uint8, error) {
	return getAsSlice[uint8](name, r, converter.DeserializeUint8Tensor)
}

func (r *InferResult) AsUint16Slice(name string) ([]uint16, error) {
	return getAsSlice[uint16](name, r, converter.DeserializeUint16Tensor)
}

func (r *InferResult) AsUint32Slice(name string) ([]uint32, error) {
	return getAsSlice[uint32](name, r, converter.DeserializeUint32Tensor)
}

func (r *InferResult) AsUint64Slice(name string) ([]uint64, error) {
	return getAsSlice[uint64](name, r, converter.DeserializeUint64Tensor)
}

func (r *InferResult) AsFloat16Slice(name string) ([]float64, error) {
	return getAsSlice[float64](name, r, converter.DeserializeFloat16Tensor)
}

func (r *InferResult) AsFloat32Slice(name string) ([]float32, error) {
	return getAsSlice[float32](name, r, converter.DeserializeFloat32Tensor)
}

func (r *InferResult) AsFloat64Slice(name string) ([]float64, error) {
	return getAsSlice[float64](name, r, converter.DeserializeFloat64Tensor)
}

func (r *InferResult) AsBoolSlice(name string) ([]bool, error) {
	return getAsSlice[bool](name, r, converter.DeserializeBoolTensor)
}

func (r *InferResult) AsByteSlice(name string) ([]string, error) {
	return getAsSlice[string](name, r, converter.DeserializeBytesTensor)
}

func getAsSlice[T any](name string, inferResult *InferResult, deserializer func(buffer []byte) ([]T, error)) ([]T, error) {
	output, err := inferResult.GetOutput(name)
	if err != nil {
		return nil, err
	}

	hasBinaryData := false
	var slice []T
	thisDataSize, ok := output.GetParameters()["binary_data_size"].(float64)
	if ok {
		hasBinaryData = true
		if thisDataSize != 0 {
			startIndex := inferResult.OutputNameToBufferMap[name]
			endIndex := startIndex + int(thisDataSize)
			dataBuffer := inferResult.Buffer[startIndex:endIndex]
			slice, err = deserializer(dataBuffer)
			if err != nil {
				return nil, err
			}
		}
	}

	if !hasBinaryData {
		slice, err = fromAnySlice[T](output.GetData())
		if err != nil {
			return nil, err
		}
	}

	return slice, nil
}

func fromAnySlice[T any](data []any) ([]T, error) {
	result := make([]T, len(data))
	for i, v := range data {
		converted, ok := v.(T)
		if !ok {
			return nil, fmt.Errorf("element at index %d cannot be converted to target type", i)
		}
		result[i] = converted
	}
	return result, nil
}
