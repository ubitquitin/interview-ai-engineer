1. Run ollama
> brew install ollama (Version MUST BE 0.20.2)
> ollama serve
> ollama run llama3.2:3b

2. Spin up docker service
> docker compose up --build
This should take ~1-2 min to build

3. Test
> install uv
> uv run tests/test_pipeline.py
This should take anywhere from 2-10 minutes depending on your system...
> uv run data/analysis.py
> pytest run...