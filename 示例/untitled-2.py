import csv

# 创建一个测试CSV文件
data = [
    ['姓名', '年龄', '城市', '职业'],
    ['张三', '25', '北京', '工程师'],
    ['李四', '30', '上海', '设计师'],
    ['王五', '28', '广州', '教师'],
    ['赵六', '35', '深圳', '医生'],
    ['钱七', '27', '杭州', '产品经理']
]

with open('test_data.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入数据
    for row in data:
        writer.writerow(row)

print("已创建测试CSV文件: test_data.csv")

# 验证文件内容
with open('test_data.csv', 'r', encoding='utf-8-sig') as csvfile:
    content = csvfile.read()
    print("\n文件内容:")
    print(content)
