# Share a machine's spare GPU (for Iana, or any teammate)

macOS or Linux. About 1 minute.

1. Install `uv` if you don't have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
   ```
2. Start sharing:
   ```bash
   uvx --from "git+https://github.com/sissississi-013/gpuroute#subdirectory=gpuroute" gpuroute share --name iana
   ```
   If macOS asks "accept incoming network connections?", click **Allow**.
3. It prints a line like this. Send it to the person running the agent:
   ```
   gpuroute join iana@172.20.x.x:7400#<token>
   ```

**What you're agreeing to:** jobs only start while your machine is mostly idle (under 60% busy), each job is capped at 30 minutes, one runs at a time, and every job is printed in your terminal with who sent it and what it installs. Jobs run as your user in `~/.gpuroute/node_jobs/`, so only share with people you trust. **Ctrl-C stops sharing immediately** and kills any running job.

Options: `--max-minutes 10`, `--idle-below 30` (stricter), `--always` (accept jobs even while busy), `--price 0.50` ($/hour, for a future marketplace).

**If the two laptops can't reach each other** (venue wifi often blocks device-to-device traffic): put both on the same phone hotspot and run step 2 again.

### Internet pool: share from anywhere (no LAN needed)

```bash
gpuroute hub     # you, once: deploys a tiny relay on your Modal account and prints the pool code
# anyone, on any Mac/Linux/NVIDIA box, anywhere:
uvx --from "git+https://github.com/sissississi-013/gpuroute#subdirectory=gpuroute" gpuroute share --pool "<pool code>" --name friend-pc
```
Nodes connect outbound over HTTPS, so venue wifi isolation and home NATs don't matter. Pool machines appear automatically in `compute_map` as **pool node**, and `run_workload(target="friend-pc")` (or `auto`) sends jobs there. The pool secret gates every request.
