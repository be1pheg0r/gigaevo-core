# Token Playground

Public, resource-bounded UI for `CostMonitorHook.project_tokens()`.

The browser can request a projection for 25–1000 mutation attempts and a
budget up to 20M tokens. Every new measurement launches exactly 10 real attempts for
one of the available AlphaEvolve tasks. The probe size, LLM configuration, Hydra options,
Redis DB and command line are not accepted from the request.

Safety limits are intentionally constants in `app.py`:

- one active probe globally;
- 10 real mutation attempts per probe;
- 15 minute hard timeout;
- one new probe per IP per 30 seconds;
- 12 new probes per process per 24 hours;
- 4 KiB request-body limit;
- 180 API requests per IP per minute and at most 500 live result handles;
- no reuse of completed measurements and `no-store` on every response;
- no experiment, log, stop, shell or generic launch routes.

Run:

```bash
cd ~/gigaevo-core
python3 tools/token_playground/app.py --selftest
setsid nohup python3 tools/token_playground/app.py > ~/token_playground.log 2>&1 < /dev/null &
```

The app binds to `127.0.0.1:8092`. Optional operator-only environment
variables are `TOKEN_PLAYGROUND_PORT` and `TOKEN_PLAYGROUND_LLM`.

Recommended nginx location:

```nginx
location /tokens/ {
    proxy_pass http://127.0.0.1:8092/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    # Overwrite the client-supplied value so the per-IP limiter cannot be spoofed.
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_connect_timeout 3s;
    proxy_read_timeout 20s;
    client_max_body_size 4k;
}
```

Do not run uvicorn with multiple workers: the global probe lock, active-job
state and rate limits are intentionally process-local.
