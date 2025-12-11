
🟦 What Is TOON?

TOON is a new data-serialization format created with one main purpose:

👉 Reduce the number of tokens when exchanging structured data with LLMs.
While JSON uses lots of quotes, braces, colons, and repeated keys…
TOON strips away all that overhead and gives you a lean and token-friendly format.

Example:

JSON:
{
  "users": [
    { "id": 1, "name": "Alice", "role": "admin" },
    { "id": 2, "name": "Bob", "role": "user" }
  ]
}

TOON:
users[2]{id,name,role}:
  1,Alice,admin
  2,Bob,user


✔ No quotes
✔ No braces
✔ No colons
✔ 30–50% fewer tokens

🚀 Why Is TOON Important Now?

LLMs like GPT, Gemini, and Claude charge per token — input + output.

That means:

More tokens = more cost

More tokens = slower responses

More tokens = less efficient prompts

If you're sending hundreds or thousands of structured items in JSON, you are wasting a LOT of tokens.

TOON fixes this by using a simple, tabular, LLM-friendly structure.

Key advantages:

✨ Up to 50% fewer tokens
✨ Easier for LLMs to reason about
✨ Better for uniform or table-like data
✨ Works in Python, JS, Go, Rust, and more

📘 JSON vs TOON (Examples)
1. Simple Object

JSON:

{ "name": "Alice", "age": 30 }


TOON:

name: Alice
age: 30

2. Array of Values

JSON:

{ "colors": ["red", "green", "blue"] }


TOON:

colors[3]: red,green,blue

3. Array of Objects

JSON:

{
  "users": [
    { "id": 1, "name": "Alice" },
    { "id": 2, "name": "Bob" }
  ]
}


TOON:

users[2]{id,name}:
  1,Alice
  2,Bob

4. Nested Objects

JSON:

{
  "user": { "id": 1, "profile": { "age": 30 } }
}


TOON:

user:
  id: 1
  profile:
    age: 30


Notice how clean that is!

💻 Using TOON in JavaScript / TypeScript

Install the official package:

npm install @toon-format/toon


Encode JSON → TOON

import { encode } from "@toon-format/toon";
console.log(encode(data));


Decode TOON → JSON

import { decode } from "@toon-format/toon";
console.log(decode(toonString));

🐍 Using TOON in Python

Install:

pip install python-toon


Encode:

from toon import encode
print(encode(data))


Decode:

from toon import decode
print(decode(toon_string))


Easy!
