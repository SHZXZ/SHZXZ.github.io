from datetime import datetime, timezone

# 1. 获取当前并包含 UTC 时区信息的时间
utc_now = datetime.now(timezone.utc)

# 2. 格式化为只包含毫秒（3位数）的 ISO 格式
#    使用 timespec='milliseconds' 限制为毫秒精度
iso_string_with_z = utc_now.isoformat(timespec='milliseconds').replace('+00:00', 'Z')

print(iso_string_with_z)
# 输出示例: 2025-07-21T10:14:01.123Z