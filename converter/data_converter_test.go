package converter

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type FaultyWriter struct {
	failOnWrite bool
}

func (fw *FaultyWriter) Write(p []byte) (int, error) {
	if fw.failOnWrite {
		return 0, errors.New("write error")
	}
	return len(p), nil
}

type FaultyStringWriter struct {
	underlyingWriter io.Writer
	failOnData       []byte
	failed           bool
}

func (fw *FaultyStringWriter) Write(p []byte) (int, error) {
	if !fw.failed && bytes.Equal(p, fw.failOnData) {
		fw.failed = true
		return 0, errors.New("write error")
	}
	return fw.underlyingWriter.Write(p)
}

func serializeInt64(data []int64) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeInt32(data []int32) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeUint16(data []uint16) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeUint32(data []uint32) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeUint64(data []uint64) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeFloat32(data []float32) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeFloat64(data []float64) []byte {
	var buf bytes.Buffer
	for _, v := range data {
		binary.Write(&buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

func serializeStrings(data []string) []byte {
	var buf bytes.Buffer
	for _, str := range data {
		strBytes := []byte(str)
		strLen := int32(len(strBytes))
		binary.Write(&buf, binary.LittleEndian, strLen)
		buf.Write(strBytes)
	}
	return buf.Bytes()
}

func TestSerializeTensor(t *testing.T) {
	testCases := []struct {
		name      string
		input     any
		expected  []byte
		expectErr bool
	}{
		{"int slice", []int{1, 2, 3}, serializeInt64([]int64{1, 2, 3}), false},
		{"int32 slice", []int32{1, 2, 3}, serializeInt32([]int32{1, 2, 3}), false},
		{"int64 slice", []int64{1, 2, 3}, serializeInt64([]int64{1, 2, 3}), false},
		{"uint16 slice", []uint16{1, 2, 3}, serializeUint16([]uint16{1, 2, 3}), false},
		{"uint32 slice", []uint32{1, 2, 3}, serializeUint32([]uint32{1, 2, 3}), false},
		{"uint64 slice", []uint64{1, 2, 3}, serializeUint64([]uint64{1, 2, 3}), false},
		{"float32 slice", []float32{1.0, 2.0, 3.0}, serializeFloat32([]float32{1.0, 2.0, 3.0}), false},
		{"float64 slice", []float64{1.0, 2.0, 3.0}, serializeFloat64([]float64{1.0, 2.0, 3.0}), false},
		{"bool slice", []bool{true, false}, []byte{1, 0}, false},
		{"byte slice", []byte{0x01, 0x02}, []byte{0x01, 0x02}, false},
		{"string slice", []string{"hello", "world"}, serializeStrings([]string{"hello", "world"}), false},
		{"unsupported type", []struct{}{}, nil, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := SerializeTensor(tc.input)

			if tc.expectErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tc.expected, result)
			}
		})
	}
}

func TestSerializeTensor_Error(t *testing.T) {

	testCases := []struct {
		description string
		input       any
		expected    []byte
		expectError bool
		errorMsg    string
	}{
		{
			description: "Serialize []int",
			input:       []int{1, 2, 3},
			expected: func() []byte {
				buf := new(bytes.Buffer)
				for _, v := range []int{1, 2, 3} {
					binary.Write(buf, binary.LittleEndian, int64(v))
				}
				return buf.Bytes()
			}(),
			expectError: false,
		},
		{
			description: "Serialize []int32",
			input:       []int32{1, 2, 3},
			expected: func() []byte {
				buf := new(bytes.Buffer)
				for _, v := range []int32{1, 2, 3} {
					binary.Write(buf, binary.LittleEndian, v)
				}
				return buf.Bytes()
			}(),
			expectError: false,
		},
		{
			description: "Unsupported tensor datatype",
			input:       []complex64{1 + 2i},
			expectError: true,
			errorMsg:    "unsupported tensor datatype",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			result, err := SerializeTensor(tc.input)
			if tc.expectError {
				if err == nil || err.Error() != tc.errorMsg {
					t.Errorf("Expected error '%v', got '%v'", tc.errorMsg, err)
				}
				return
			}
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if !bytes.Equal(result, tc.expected) {
				t.Errorf("Expected %v, got %v", tc.expected, result)
			}
		})
	}

	t.Run("Simulate write error int", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]int{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error int32", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]int32{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error int64", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]int64{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error uint16", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]uint16{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error uint32", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]uint32{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error uint64", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]uint64{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error float32", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]float32{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error float64", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]float64{1, 2, 3}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error bool", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]bool{true, false}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error byte", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]byte{1, 4, 5}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error string", func(t *testing.T) {
		faultyWriter := &FaultyWriter{failOnWrite: true}
		err := serializeTensorToWriter([]string{"test1", "test2"}, faultyWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})

	t.Run("Simulate write error with []string", func(t *testing.T) {
		str := "errorString"
		tensor := []string{str, "test"}

		buffer := &bytes.Buffer{}
		failingWriter := &FaultyStringWriter{
			underlyingWriter: buffer,
			failOnData:       []byte(str),
		}

		err := serializeTensorToWriter(tensor, failingWriter)
		if err == nil || err.Error() != "write error" {
			t.Errorf("Expected 'write error', got '%v'", err)
		}
	})
}

func TestFlattenData(t *testing.T) {
	testCases := []struct {
		name     string
		input    any
		expected []any
	}{
		{"int slice", []int{1, 2, 3}, []any{1, 2, 3}},
		{"int32 slice", []int32{1, 2, 3}, []any{int32(1), int32(2), int32(3)}},
		{"int64 slice", []int64{1, 2, 3}, []any{int64(1), int64(2), int64(3)}},
		{"uint16 slice", []uint16{1, 2, 3}, []any{uint16(1), uint16(2), uint16(3)}},
		{"uint32 slice", []uint32{1, 2, 3}, []any{uint32(1), uint32(2), uint32(3)}},
		{"uint64 slice", []uint64{1, 2, 3}, []any{uint64(1), uint64(2), uint64(3)}},
		{"float32 slice", []float32{1.0, 2.0, 3.0}, []any{float32(1.0), float32(2.0), float32(3.0)}},
		{"float64 slice", []float64{1.0, 2.0, 3.0}, []any{float64(1.0), float64(2.0), float64(3.0)}},
		{"byte slice", []byte{0x01, 0x02}, []any{byte(0x01), byte(0x02)}},
		{"bool slice", []bool{true, false}, []any{true, false}},
		{"string slice", []string{"hello", "world"}, []any{"hello", "world"}},
		{"unsupported type", []struct{}{}, nil},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := FlattenData(tc.input)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestDeserializeInt8Tensor(t *testing.T) {
	data := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	expected := []int8{1, 2, 3, 4, 5, 6, 7, 8}

	result, err := DeserializeNumericSlice[int8](data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestDeserializeInt16Tensor(t *testing.T) {
	data := []byte{1, 1}
	expected := []int16{257}

	result, err := DeserializeNumericSlice[int16](data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestDeserializeInt32Tensor(t *testing.T) {
	data := []byte{1, 1, 0, 0}
	expected := []int32{257}

	result, err := DeserializeNumericSlice[int32](data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestDeserializeInt64Tensor(t *testing.T) {

	data := []byte{1, 1, 0, 0, 0, 0, 0, 0}
	expected := []int64{257}
	result, err := DeserializeNumericSlice[int64](data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("DeserializeInt64Tensor(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("DeserializeInt64Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestDeserializeUint8Tensor(t *testing.T) {

	data := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	expected := []uint8{1, 2, 3, 4, 5, 6, 7, 8}
	result, err := DeserializeNumericSlice[uint8](data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("DeserializeUint8Tensor(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("DeserializeUint8Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestDeserializeUint16Tensor(t *testing.T) {

	data := []byte{1, 1}
	expected := []uint16{257}
	result, err := DeserializeNumericSlice[uint16](data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("DeserializeUint16Tensor(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("DeserializeUint16Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestDeserializeUint32Tensor(t *testing.T) {

	data := []byte{1, 1, 0, 0}
	expected := []uint32{257}
	result, err := DeserializeNumericSlice[uint32](data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("DeserializeUint32Tensor(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("DeserializeUint32Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestDeserializeUint64Tensor(t *testing.T) {

	data := []byte{1, 1, 0, 0, 0, 0, 0, 0}
	expected := []uint64{257}
	result, err := DeserializeNumericSlice[uint64](data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("DeserializeUint64Tensor(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("DeserializeUint64Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestDeserializeFloat16Tensor(t *testing.T) {

	data := []byte{141, 122}
	expected := []float64{53664}
	result, err := DeserializeFloat16Tensor(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("DeserializeFloat16Tensor(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("DeserializeFloat16Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestDeserializeFloat32Tensor(t *testing.T) {

	data := make([]byte, 4)
	binary.LittleEndian.PutUint32(data, math.Float32bits(257.0))
	expected := []float32{257}
	result, err := DeserializeNumericSlice[float32](data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("DeserializeFloat32Tensor(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("DeserializeFloat32Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestDeserializeBF16Tensor(t *testing.T) {

	data := make([]byte, 4)
	binary.LittleEndian.PutUint32(data, math.Float32bits(256.0))
	expected := []float32{0, 256}
	result, err := DeserializeBF16Tensor(data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("DeserializeFloat32Tensor(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("DeserializeFloat32Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestDeserializeFloat64Tensor(t *testing.T) {

	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, math.Float64bits(257.0))
	expected := []float64{257}
	result, err := DeserializeNumericSlice[float64](data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("DeserializeFloat64Tensor(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("DeserializeFloat64Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestDeserializeBoolTensor(t *testing.T) {
	data := []byte{1}
	expected := []bool{true}

	result, err := DeserializeBoolTensor(data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestDeserializeBoolTensor_Comprehensive(t *testing.T) {
	testCases := []struct {
		name     string
		data     []byte
		expected []bool
	}{
		{
			name:     "empty buffer",
			data:     []byte{},
			expected: []bool{},
		},
		{
			name:     "single true",
			data:     []byte{1},
			expected: []bool{true},
		},
		{
			name:     "single false",
			data:     []byte{0},
			expected: []bool{false},
		},
		{
			name:     "multiple values",
			data:     []byte{1, 0, 255, 2, 0},
			expected: []bool{true, false, true, true, false},
		},
		{
			name:     "all non-zero as true",
			data:     []byte{1, 2, 3, 255},
			expected: []bool{true, true, true, true},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := DeserializeBoolTensor(tc.data)

			require.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestDeserializeFloat16Tensor_EdgeCases(t *testing.T) {
	t.Run("empty buffer", func(t *testing.T) {
		result, err := DeserializeFloat16Tensor([]byte{})

		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("odd length buffer", func(t *testing.T) {
		result, err := DeserializeFloat16Tensor([]byte{1, 2, 3})

		assert.Error(t, err)
		assert.EqualError(t, err, "data buffer length (3) is not a multiple of 2")
		assert.Nil(t, result)
	})

	t.Run("valid even length", func(t *testing.T) {
		data := []byte{1, 2, 3, 4}
		result, err := DeserializeFloat16Tensor(data)

		require.NoError(t, err)
		assert.Len(t, result, len(data)/2)
	})
}

func TestDeserializeBF16Tensor_EdgeCases(t *testing.T) {
	t.Run("empty buffer", func(t *testing.T) {
		result, err := DeserializeBF16Tensor([]byte{})

		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("odd length buffer", func(t *testing.T) {
		result, err := DeserializeBF16Tensor([]byte{1, 2, 3})

		assert.Error(t, err)
		assert.EqualError(t, err, "data buffer length (3) is not a multiple of 2")
		assert.Nil(t, result)
	})

	t.Run("valid even length", func(t *testing.T) {
		data := []byte{1, 2, 3, 4}
		result, err := DeserializeBF16Tensor(data)

		require.NoError(t, err)
		assert.Len(t, result, len(data)/2)
	})
}

func TestDeserializeBytesTensor(t *testing.T) {

	data := serializeStrings([]string{"hello", "world"})
	result, err := DeserializeBytesTensor(data)
	expected := []string{"hello", "world"}
	if err != nil {
		t.Errorf("deserializeBytesTensor(%v) unexpected error: %v", data, err)
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("deserializeBytesTensor(%v) = %v; want %v", data, result, expected)
	}
	data = []byte{0x05, 0x00, 0x00}
	_, err = DeserializeBytesTensor(data)
	if err == nil {
		t.Errorf("Expected error for malformed data")
	}
}

func TestDeserializeBytesTensor_UnexpectedEnd(t *testing.T) {

	length := uint32(5)
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, length)
	buf = append(buf, []byte("abc")...)

	_, err := DeserializeBytesTensor(buf)
	if err == nil || !strings.HasPrefix(err.Error(), "unexpected end of tensor") {
		t.Errorf("Expected 'unexpected end of tensor' prefix, got '%v'", err)
	}
}

func TestConvertByteSliceToInt64Slice(t *testing.T) {

	data := make([]byte, 16)
	binary.LittleEndian.PutUint64(data[0:], uint64(1))
	binary.LittleEndian.PutUint64(data[8:], uint64(2))
	expected := []int64{1, 2}
	result, err := DeserializeNumericSlice[int64](data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertByteSliceToInt64Slice(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("ConvertByteSliceToInt64Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestConvertByteSliceToFloat32Slice(t *testing.T) {

	data := make([]byte, 8)
	binary.LittleEndian.PutUint32(data[0:], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(data[4:], math.Float32bits(2.0))
	expected := []float32{1.0, 2.0}
	result, err := DeserializeNumericSlice[float32](data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertByteSliceToFloat32Slice(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("ConvertByteSliceToFloat32Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestConvertByteSliceToFloat64Slice(t *testing.T) {

	data := make([]byte, 16)
	binary.LittleEndian.PutUint64(data[0:], math.Float64bits(1.0))
	binary.LittleEndian.PutUint64(data[8:], math.Float64bits(2.0))
	expected := []float64{1.0, 2.0}
	result, err := DeserializeNumericSlice[float64](data)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("ConvertByteSliceToFloat64Slice(%v) = %v; want %v", data, result, expected)
	}
	if err != nil {
		t.Errorf("ConvertByteSliceToFloat64Tensor(%v) = %v; want %v", data, err, nil)
	}
}

func TestConvertToFloat32(t *testing.T) {
	data := []any{float64(1.0), float64(2.0)}
	expected := []any{float32(1.0), float32(2.0)}

	result, err := ConvertToNumericSlice[float32](data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestConvertToFloat64(t *testing.T) {
	data := []any{float64(1.0), float64(2.0)}
	expected := []any{float64(1.0), float64(2.0)}

	result, err := ConvertToNumericSlice[float64](data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestConvertToInt32(t *testing.T) {
	data := []any{float64(1), float64(2)}
	expected := []any{int32(1), int32(2)}

	result, err := ConvertToNumericSlice[int32](data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestConvertToInt64(t *testing.T) {
	data := []any{float64(1), float64(2)}
	expected := []any{int64(1), int64(2)}

	result, err := ConvertToNumericSlice[int64](data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestConvertToUint32(t *testing.T) {
	data := []any{float64(1), float64(2)}
	expected := []any{uint32(1), uint32(2)}

	result, err := ConvertToNumericSlice[uint32](data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestConvertToUint64(t *testing.T) {
	data := []any{float64(1), float64(2)}
	expected := []any{uint64(1), uint64(2)}

	result, err := ConvertToNumericSlice[uint64](data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestConvertToBool(t *testing.T) {
	data := []any{true, false}
	expected := []bool{true, false}

	result, err := ConvertToBoolSlice(data)

	require.NoError(t, err)
	assert.Equal(t, expected, result)
}

func TestConvertToBytes(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		data := []any{"hello", "world"}
		expected := [][]byte{[]byte("hello"), []byte("world")}

		result, err := ConvertToBytesSlice(data)

		require.NoError(t, err)
		assert.Equal(t, expected, result)
	})

	t.Run("error with non-string", func(t *testing.T) {
		data := []any{"hello", 123}

		_, err := ConvertToBytesSlice(data)

		assert.Error(t, err)
	})
}

func TestReshape1D(t *testing.T) {
	data := []int{1, 2, 3, 4, 5}
	shape := []int64{5}
	reshaped, err := Reshape1D(data, shape)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(reshaped, data) {
		t.Errorf("expected %v, got %v", data, reshaped)
	}

	badShape := []int64{2}
	_, err = Reshape1D(data, badShape)
	if err == nil {
		t.Error("expected error for shape with dimensions != 1, got nil")
	}

	badShape2 := []int64{6}
	_, err = Reshape1D(data, badShape2)
	if err == nil {
		t.Error("expected error for data length mismatch, got nil")
	}
}

func TestReshape2D(t *testing.T) {
	data := []int{1, 2, 3, 4, 5, 6}
	shape := []int64{2, 3}
	expected := [][]int{
		{1, 2, 3},
		{4, 5, 6},
	}
	reshaped, err := Reshape2D(data, shape)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(reshaped, expected) {
		t.Errorf("expected %v, got %v", expected, reshaped)
	}

	badShape := []int64{3}
	_, err = Reshape2D(data, badShape)
	if err == nil {
		t.Error("expected error for shape with dimensions != 2, got nil")
	}

	badShape2 := []int64{3, 3}
	_, err = Reshape2D(data, badShape2)
	if err == nil {
		t.Error("expected error for data length mismatch, got nil")
	}
}

func TestReshape3D(t *testing.T) {

	t.Run("success", func(t *testing.T) {
		data := []int{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		}
		shape := []int64{2, 2, 3}
		expected := [][][]int{
			{
				{1, 2, 3},
				{4, 5, 6},
			},
			{
				{7, 8, 9},
				{10, 11, 12},
			},
		}
		reshaped, err := Reshape3D(data, shape)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(reshaped, expected) {
			t.Errorf("expected %v, got %v", expected, reshaped)
		}

		badShape := []int64{2, 3}
		_, err = Reshape3D(data, badShape)
		if err == nil {
			t.Error("expected error for shape with dimensions != 3, got nil")
		}

		badShape2 := []int64{2, 2, 4}
		_, err = Reshape3D(data, badShape2)
		if err == nil {
			t.Error("expected error for data length mismatch, got nil")
		}
	})

	// Test Case 1: Successful reshaping of an integer slice.
	t.Run("SuccessfulReshapeInts", func(t *testing.T) {
		data := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		shape := []int64{2, 3, 2}
		expected := [][][]int{
			{{1, 2}, {3, 4}, {5, 6}},
			{{7, 8}, {9, 10}, {11, 12}},
		}

		result, err := Reshape3D(data, shape)
		if err != nil {
			t.Fatalf("Expected no error, but got: %v", err)
		}
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("Result does not match expected value.\nGot: %v\nWant: %v", result, expected)
		}
	})

	// Test Case 2: Successful reshaping of a string slice to test generics.
	t.Run("SuccessfulReshapeStrings", func(t *testing.T) {
		data := []string{"a", "b", "c", "d"}
		shape := []int64{2, 2, 1}
		expected := [][][]string{
			{{"a"}, {"b"}},
			{{"c"}, {"d"}},
		}

		result, err := Reshape3D(data, shape)
		if err != nil {
			t.Fatalf("Expected no error, but got: %v", err)
		}
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("Result does not match expected value.\nGot: %v\nWant: %v", result, expected)
		}
	})

	// Test Case 3: Error case for incorrect shape dimensions.
	t.Run("ErrorIncorrectShapeDimensions", func(t *testing.T) {
		data := []int{1, 2, 3, 4}
		shape := []int64{2, 2} // Only 2 dimensions provided

		result, err := Reshape3D(data, shape)
		if err == nil {
			t.Fatal("Expected an error for incorrect shape dimensions, but got nil")
		}
		if result != nil {
			t.Errorf("Expected result to be nil on error, but got: %v", result)
		}
	})

	// Test Case 4: Error case for data length mismatch.
	t.Run("ErrorDataLengthMismatch", func(t *testing.T) {
		data := []int{1, 2, 3, 4, 5} // 5 elements
		shape := []int64{2, 2, 1}    // Expects 4 elements

		result, err := Reshape3D(data, shape)
		if err == nil {
			t.Fatal("Expected an error for data length mismatch, but got nil")
		}
		if result != nil {
			t.Errorf("Expected result to be nil on error, but got: %v", result)
		}
	})

	// Test Case 5: Edge case with empty data and zero-sized shape.
	t.Run("EdgeCaseEmptyDataAndShape", func(t *testing.T) {
		var data []float64 // Empty slice
		shape := []int64{0, 10, 10}
		expected := [][][]float64{}

		result, err := Reshape3D(data, shape)
		if err != nil {
			t.Fatalf("Expected no error for empty data, but got: %v", err)
		}
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("Result does not match expected value for empty data.\nGot: %v\nWant: %v", result, expected)
		}
	})

	// Test Case 6: Edge case where one dimension is 1.
	t.Run("EdgeCaseOneDimensionIsOne", func(t *testing.T) {
		data := []int{1, 2, 3, 4}
		shape := []int64{4, 1, 1}
		expected := [][][]int{
			{{1}},
			{{2}},
			{{3}},
			{{4}},
		}

		result, err := Reshape3D(data, shape)
		if err != nil {
			t.Fatalf("Expected no error, but got: %v", err)
		}
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("Result does not match expected value.\nGot: %v\nWant: %v", result, expected)
		}
	})
}
