import json

d = {
    "name" : "你是",
    "age" : 18,
    "gender" : "male"
}

s = json.dumps(d)  # 转成json格式的
print(s)