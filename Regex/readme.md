# 🧭 LỘ TRÌNH HỌC REGEX101 – FROM ZERO → PRO CƠ BẢN

👉 Mở: [https://regex101.com](https://regex101.com)

## 🧩 Giao diện regex101 — hiểu trước đã

Có 3 vùng chính:

```
┌────────────────────────────┐
│  REGEX INPUT               │  ← m viết regex ở đây
├────────────────────────────┤
│  TEST STRING               │  ← m paste text test ở đây
├────────────────────────────┤
│  EXPLANATION / MATCHES     │  ← tool giải thích
└────────────────────────────┘
```

⚙️ **Chọn Flavor:** Python (bên trái)

## 🟢 BÀI 1 — Match số bất kỳ

**Regex:**
```
\d+
```

**Test string:**
```
abc123 xyz45 pqr9
```

👉 **Kết quả:**
- `123`
- `45`
- `9`

💡 **Giải thích:**
- `\d` = digit
- `+` = 1 → vô hạn

## 🟢 BÀI 2 — Match đúng 10 số liên tiếp

**Regex:**
```
\d{10}
```

**Test:**
```
0987654321
01234567890
```

👉 **Match:**
- `0987654321`
- `0123456789`

💡 `{10}` = đúng 10 ký tự

## 🟢 BÀI 3 — Validate toàn chuỗi (neo đầu + cuối)

**Regex:**
```
^\d{10}$
```

**Test:**
- `0987654321` ✅
- `abc0987654321` ❌
- `0987654321xyz` ❌

💡 **Giải thích:**
- `^` = start
- `$` = end
- → Không cho dư thừa

## 🟡 BÀI 4 — Match chữ cái

**Regex:**
```
[a-z]+
```

**Test:**
```
abc XYZ hello123
```

👉 **Match:**
- `abc`
- `hello`

## 🟡 BÀI 5 — Match chữ + số

**Regex:**
```
[a-zA-Z0-9]+
```

**Test:**
```
abc123 XYZ_456
```

👉 **Match:**
- `abc123`
- `XYZ`
- `456`

## 🟠 BÀI 6 — Group & Capture

**Regex:**
```
(\d{2})([A-Z])-(\d{4,5})
```

**Test:**
```
29A-12345
```

👉 **Groups:**
1. `29`
2. `A`
3. `12345`

🔥 **Đây là nền tảng của extract data**

## 🔴 BÀI 7 — Validate biển số xe VN

**Regex:**
```
^\d{2}[A-Z]-\d{4,5}$
```

**Test:**
- `29A-12345` ✅
- `30F-9999` ✅
- `1A-12345` ❌
- `29AA-1234` ❌

🔥 **Đây là regex m sẽ xài liên tục trong ALPR**

## 🧠 Tư duy build regex (quan trọng nhất)

Muốn viết regex, đừng gõ bừa — hãy:

**B1. Nhìn format**
```
29A-12345
```

**B2. Chia nhỏ**
- `29` → `\d{2}`
- `A` → `[A-Z]`
- `-` → `-`
- `12345` → `\d{4,5}`

**B3. Ghép lại**
```
^\d{2}[A-Z]-\d{4,5}$
```

## ⚡ Trick dùng regex101 cho nhanh

1. Hover từng token → đọc giải thích
2. Nhìn bảng Explanation

→ đọc như trace code 😈

## 🧪 Mini challenge cho m (làm trên regex101)

### 1. Match email
```
abc@gmail.com
sv123@tlu.edu.vn
x@y.z
```

👉 Viết regex match hết.

### 2. Match datetime
```
2025-01-24
2026-12-31
```

### 3. Match plate sai OCR
```
29A-1234S
30F-999O
```

👉 Regex phát hiện chứa ký tự dễ nhầm OCR (S,O,I,Z)

## 🔥 Sau khi xong regex101 → sang Python

Lúc đó regex sẽ:

- Không còn là ký hiệu lạ
- Mà là tool cực mạnh để lọc OCR noise
