"""Check if DeepSeek V4 Pro exposes reasoning_content in the response."""
import json, sys
from pathlib import Path
from openai import OpenAI
from src.config import DS_API_KEY, MODEL_NAME, API_BASE_URL
from src.utils import load_json

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

q = load_json("data/browsecomp-zh-decrypted.json")[6]
question = q["Question"]

# Use V1 prompt (entity hypotheses)
v1_prompt = f"""你是一个知识渊博的推理助手。请根据以下谜题的约束条件，推测谜题中涉及的可搜索实体名称。

重要：输出的是谜题的中间实体（物质名、人名、地名、作品名等具体名词），而不是最终答案。
- 如果问题问"哪个年份"，应输出该事件涉及的具体物质/人物名称，而非年份
- 如果问题问"哪个朝代"，应输出该事物/人物的具体名称，而非朝代名
- 如果问题问"哪家公司"，应输出创始人的名字或相关产品名，而非公司名

谜题：{question}

输出一个JSON字符串数组，包含5个可搜索的中间实体名称，覆盖不同方向。只输出JSON数组，不要其他文字。

示例：["青霉素","链霉素","四环素","磺胺","红霉素"]

JSON数组："""

client = OpenAI(api_key=DS_API_KEY, base_url=API_BASE_URL, timeout=300.0)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": v1_prompt}],
    temperature=0.0,
    max_tokens=65536,
)

msg = response.choices[0].message
print("=== MESSAGE FIELDS ===")
print(f"content: {msg.content}")
print(f"\n--- All attributes ---")
for attr in dir(msg):
    if not attr.startswith('_'):
        try:
            val = getattr(msg, attr)
            if val is not None:
                s = str(val)
                print(f"  {attr}: {s[:500]}")
        except:
            pass

print(f"\n=== USAGE ===")
if response.usage:
    print(f"  prompt_tokens: {response.usage.prompt_tokens}")
    print(f"  completion_tokens: {response.usage.completion_tokens}")
    print(f"  total_tokens: {response.usage.total_tokens}")
    details = response.usage.completion_tokens_details
    if details:
        print(f"  reasoning_tokens: {details.reasoning_tokens}")
        for attr in dir(details):
            if not attr.startswith('_'):
                try:
                    val = getattr(details, attr)
                    if val is not None:
                        print(f"  details.{attr}: {val}")
                except:
                    pass

# Also check raw response
print(f"\n=== RAW RESPONSE (model dump) ===")
print(response.model_dump_json(indent=2)[:3000])
