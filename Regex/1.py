import re
text = "My phone: 0987654321"
m = re.search(r"\d{10}", text)

# if m:
#     print(m.group())


print(m)