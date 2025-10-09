.PHONY: generate-mocks

generate-mocks:
	mockgen -source=base/http_client.go -destination=mocks/mock_http_client.go -package=mocks
	mockgen -source=base/client.go -destination=base/mock_client.go -package=base
	mockgen -source=client/grpc/grpc_generated_v2/grpc_service.pb.go -destination=mocks/mock_grpc_service.go -package=mocks
	mockgen -source=client/grpc/grpc_generated_v2/grpc_service_grpc.pb.go -destination=mocks/mock_grpc_service_grpc.go -package=mocks
	mockgen -source=base/infer_input.go -destination=mocks/mock_infer_input.go -package=mocks
	mockgen -source=base/infer_output.go -destination=mocks/mock_infer_output.go -package=mocks
	mockgen -source=client/grpc/grpc_generated_v2/model_config.pb.go -destination=mocks/mock_model_config.go -package=mocks
	mockgen -source=base/response_wrapper.go -destination=mocks/mock_response_wrapper.go -package=mocks
	mockgen -source=base/infer_result.go -destination=base/mock_infer_result.go -package=base
	