import pytest

from deepspeedometer import parse_args_to_configs, BenchmarkRunner


@pytest.mark.parametrize("num_clients", [(1, 2, 4)], indirect=True)
def test_early_stop(benchmark_args):
    benchmark_args += [
        "--early_stop_latency",
        "1",
        "--dummy_client_latency_time",
        "2.0",
    ]
    print(benchmark_args)
    benchmark_config, client_config = parse_args_to_configs(benchmark_args)
    benchmark_runner = BenchmarkRunner(benchmark_config, client_config)
    benchmark_runner.run()

    expected_results = 1
    actual_results = len(list(benchmark_runner._get_output_dir().glob("*.json")))
    assert (
        expected_results == actual_results
    ), f"Number of result files ({actual_results}) does not match expected number ({expected_results})."
