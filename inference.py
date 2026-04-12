import asyncio
import json
import os
import re
from typing import List, Optional
from openai import OpenAI
from client import TicketRouterEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "ticket_router"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rstr}", flush=True)

def call_llm(client: OpenAI, obs_dict: dict) -> dict:
    sys_msg = "You are an AI support bot. Output valid JSON only: {\"act_type\": \"search\", \"query\": \"<text>\"} OR {\"act_type\": \"route\", \"dept\": \"<dept>\"}"
    user_msg = json.dumps(obs_dict)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0
        )
        txt = (resp.choices[0].message.content or "").strip()
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
        return json.loads(txt)
    except Exception:
        return {"act_type": "route", "dept": "Tech Support"}

async def run_ep(client: OpenAI, env: TicketRouterEnv, diff: str) -> float:
    task_id = f"triage_{diff}_001"
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    rewards: List[float] = []
    step_count = 0
    score = 0.0
    
    try:
        res = await env.reset(difficulty=diff)
        obs_dict = res.observation
        
        done = False
        while not done and step_count < 5:
            step_count += 1
            act_dict = call_llm(client, obs_dict)
            act_str = json.dumps(act_dict)
            
            res = await env.step(action=act_dict)
            obs_dict = res.observation
            reward = float(res.reward if res.reward is not None else 0.0)
            done = bool(res.done)
            err = obs_dict.get("error", None)
            
            rewards.append(reward)
            log_step(step=step_count, action=act_str, reward=reward, done=done, error=err)
            
            if done:
                score = reward
                
    except Exception as e:
        log_step(step=step_count+1, action="error", reward=0.0, done=True, error=str(e))
        rewards.append(0.0)
        step_count += 1
    
    score = max(0.01, min(score, 0.99))
    rewards = [max(0.01, min(r, 0.99)) for r in rewards]
    success = score >= 0.5
    log_end(success=success, steps=step_count, score=score, rewards=rewards)
    return score

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    if LOCAL_IMAGE_NAME:
        env = await TicketRouterEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        base = os.getenv("TICKET_ROUTER_URL", "http://localhost:7860")
        env = TicketRouterEnv(base_url=base)
        await env.__aenter__()

    try:
        for d in ["easy", "medium", "hard"]:
            await run_ep(client, env, d)
    finally:
        try:
            await env.close()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
