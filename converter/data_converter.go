package converter

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/x448/float16"
	"io"
	"math"
)

func SerializeTensor(inputTensor any) ([]byte, error) {
	var buffer bytes.Buffer
	if err := serializeTensorToWriter(inputTensor, &buffer); err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}

// serializeTensorToWriter writes the tensor data into the provided io.Writer.
func serializeTensorToWriter(inputTensor any, w io.Writer) error {
	switch tensor := inputTensor.(type) {
	case []int:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, int64(v)); err != nil {
				return err
			}
		}
	case []int32:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []int64:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []uint16:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []uint32:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []uint64:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []float32:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []float64:
		for _, v := range tensor {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	case []bool:
		for _, v := range tensor {
			var boolVal byte
			if v {
				boolVal = 1
			}
			if err := binary.Write(w, binary.LittleEndian, boolVal); err != nil {
				return err
			}
		}
	case []byte:
		if _, err := w.Write(tensor); err != nil {
			return err
		}
	case []string:
		for _, str := range tensor {
			strBytes := []byte(str)
			strLen := int32(len(strBytes))
			if err := binary.Write(w, binary.LittleEndian, strLen); err != nil {
				return err
			}
			if _, err := w.Write(strBytes); err != nil {
				return err
			}
		}
	default:
		return errors.New("unsupported tensor datatype")
	}
	return nil
}

func FlattenData(inputTensor any) []any {
	switch tensor := inputTensor.(type) {
	case []int:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []int32:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []int64:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []uint16:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []uint32:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []uint64:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []float32:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []float64:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []byte:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []bool:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	case []string:
		result := make([]any, len(tensor))
		for i, v := range tensor {
			result[i] = v
		}
		return result
	default:
		return nil
	}
}

func DeserializeInt8Tensor(dataBuffer []byte) ([]int8, error) {
	return deserializeTensorGeneric(dataBuffer, 1, func(b []byte) int8 { return int8(b[0]) }), nil
}

func DeserializeInt16Tensor(dataBuffer []byte) ([]int16, error) {
	return deserializeTensorGeneric(dataBuffer, 2, func(b []byte) int16 {
		return int16(binary.LittleEndian.Uint16(b))
	}), nil
}

func DeserializeInt32Tensor(dataBuffer []byte) ([]int32, error) {
	return deserializeTensorGeneric(dataBuffer, 4, func(b []byte) int32 {
		return int32(binary.LittleEndian.Uint32(b))
	}), nil
}

func DeserializeInt64Tensor(dataBuffer []byte) ([]int64, error) {
	return deserializeTensorGeneric(dataBuffer, 8, func(b []byte) int64 {
		return int64(binary.LittleEndian.Uint64(b))
	}), nil
}

func DeserializeUint8Tensor(dataBuffer []byte) ([]uint8, error) {
	// []byte is already []uint8.
	return dataBuffer, nil
}

func DeserializeUint16Tensor(dataBuffer []byte) ([]uint16, error) {
	return deserializeTensorGeneric(dataBuffer, 2, func(b []byte) uint16 {
		return binary.LittleEndian.Uint16(b)
	}), nil
}

func DeserializeUint32Tensor(dataBuffer []byte) ([]uint32, error) {
	return deserializeTensorGeneric(dataBuffer, 4, func(b []byte) uint32 {
		return binary.LittleEndian.Uint32(b)
	}), nil
}

func DeserializeUint64Tensor(dataBuffer []byte) ([]uint64, error) {
	return deserializeTensorGeneric(dataBuffer, 8, func(b []byte) uint64 {
		return binary.LittleEndian.Uint64(b)
	}), nil
}

func DeserializeBoolTensor(dataBuffer []byte) ([]bool, error) {
	return deserializeTensorGeneric(dataBuffer, 1, func(b []byte) bool {
		return b[0] != 0
	}), nil
}

func DeserializeFloat16Tensor(dataBuffer []byte) ([]float64, error) {
	return deserializeTensorGeneric(dataBuffer, 2, func(b []byte) float64 {
		uint16Value := binary.LittleEndian.Uint16(b)
		float16Value := float16.Frombits(uint16Value)
		return float64(float16Value.Float32())
	}), nil
}

func DeserializeFloat32Tensor(dataBuffer []byte) ([]float32, error) {
	return deserializeTensorGeneric(dataBuffer, 4, func(b []byte) float32 {
		return math.Float32frombits(binary.LittleEndian.Uint32(b))
	}), nil
}

func DeserializeFloat64Tensor(dataBuffer []byte) ([]float64, error) {
	return deserializeTensorGeneric(dataBuffer, 8, func(b []byte) float64 {
		return math.Float64frombits(binary.LittleEndian.Uint64(b))
	}), nil
}

func DeserializeBF16Tensor(encodedTensor []byte) ([]float32, error) {
	return deserializeTensorGeneric(encodedTensor, 2, func(b []byte) float32 {
		bits := binary.LittleEndian.Uint16(b)
		float32Bits := uint32(bits) << 16
		return math.Float32frombits(float32Bits)
	}), nil
}

func DeserializeBytesTensor(encodedTensor []byte) ([]string, error) {
	var strs []string
	offset := 0
	for offset < len(encodedTensor) {
		if offset+4 > len(encodedTensor) {
			return nil, fmt.Errorf("unexpected end of encoded tensor")
		}
		length := binary.LittleEndian.Uint32(encodedTensor[offset : offset+4])
		offset += 4
		if offset+int(length) > len(encodedTensor) {
			return nil, fmt.Errorf("unexpected end of encoded tensor")
		}
		strs = append(strs, string(encodedTensor[offset:offset+int(length)]))
		offset += int(length)
	}
	return strs, nil
}

// Helper: deserializeTensorGeneric splits dataBuffer into blocks of blockSize and converts each block using convert.
func deserializeTensorGeneric[T any](dataBuffer []byte, blockSize int, convert func([]byte) T) []T {
	n := len(dataBuffer) / blockSize
	result := make([]T, n)
	for i := 0; i < n; i++ {
		start := i * blockSize
		end := start + blockSize
		result[i] = convert(dataBuffer[start:end])
	}
	return result
}

func ConvertInterfaceSliceToFloat32SliceAsInterface(data []any) []any {
	return convertInterfaceSlice(data, func(v any) any { return float32(v.(float64)) })
}

func ConvertInterfaceSliceToFloat64SliceAsInterface(data []any) []any {
	return convertInterfaceSlice(data, func(v any) any { return v.(float64) })
}

func ConvertInterfaceSliceToInt32SliceAsInterface(data []any) []any {
	return convertInterfaceSlice(data, func(v any) any { return int32(v.(float64)) })
}

func ConvertInterfaceSliceToInt64SliceAsInterface(data []any) []any {
	return convertInterfaceSlice(data, func(v any) any { return int64(v.(float64)) })
}

func ConvertInterfaceSliceToUint32SliceAsInterface(data []any) []any {
	return convertInterfaceSlice(data, func(v any) any { return uint32(v.(float64)) })
}

func ConvertInterfaceSliceToUint64SliceAsInterface(data []any) []any {
	return convertInterfaceSlice(data, func(v any) any { return uint64(v.(float64)) })
}

func ConvertInterfaceSliceToBoolSliceAsInterface(data []any) []any {
	return convertInterfaceSlice(data, func(v any) any { return v.(bool) })
}

func ConvertInterfaceSliceToBytesSliceAsInterface(data []any) ([]any, error) {
	convertedData := make([]any, len(data))
	for i, v := range data {
		str, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("expected BYTES datatype, got %T", v)
		}
		convertedData[i] = []byte(str)
	}
	return convertedData, nil
}

// convertInterfaceSlice is a helper to convert []any to another []any using a converter function.
func convertInterfaceSlice(data []any, conv func(any) any) []any {
	result := make([]any, len(data))
	for i, v := range data {
		result[i] = conv(v)
	}
	return result
}

// Reshape1D converts a flat []T into a 1D slice according to the provided shape.
func Reshape1D[T any](data []T, shape []int64) ([]T, error) {
	if len(shape) != 1 {
		return nil, fmt.Errorf("expected 1D shape, got %d dimensions", len(shape))
	}
	total := int(shape[0])
	if len(data) != total {
		return nil, fmt.Errorf("data length mismatch: expected %d, got %d", total, len(data))
	}
	return data, nil
}

// Reshape2D converts a flat []T into a 2D slice ([][]T) according to the provided shape.
func Reshape2D[T any](data []T, shape []int64) ([][]T, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("expected 2D shape, got %d dimensions", len(shape))
	}
	rows := int(shape[0])
	cols := int(shape[1])
	if len(data) != rows*cols {
		return nil, fmt.Errorf("data length mismatch: expected %d, got %d", rows*cols, len(data))
	}
	res := make([][]T, rows)
	for i := 0; i < rows; i++ {
		res[i] = data[i*cols : (i+1)*cols]
	}
	return res, nil
}

// Reshape3D converts a flat []T into a 3D slice ([][][]T) according to the provided shape.
func Reshape3D[T any](data []T, shape []int64) ([][][]T, error) {
	if len(shape) != 3 {
		return nil, fmt.Errorf("expected 3D shape, got %d dimensions", len(shape))
	}
	d0, d1, d2 := int(shape[0]), int(shape[1]), int(shape[2])
	if len(data) != d0*d1*d2 {
		return nil, fmt.Errorf("data length mismatch: expected %d, got %d", d0*d1*d2, len(data))
	}
	res := make([][][]T, d0)
	for i := 0; i < d0; i++ {
		res[i] = make([][]T, d1)
		for j := 0; j < d1; j++ {
			start := (i*d1 + j) * d2
			res[i][j] = data[start : start+d2]
		}
	}
	return res, nil
}
