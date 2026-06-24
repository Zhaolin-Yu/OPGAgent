#!/usr/bin/env python3
"""
生成所有模型需要的类别映射文件
"""

import json
import os

# 32 牙齿 ID
tooth_32_categories = {
    "categories": [
        {"id": i+1, "name": tooth_id} 
        for i, tooth_id in enumerate([
            "11", "21", "22", "23", "12", "13", "15", "17",
            "27", "28", "38", "36", "34", "33", "32", "31",
            "41", "43", "44", "45", "48", "18", "16", "42",
            "35", "47", "24", "25", "14", "26", "37", "46"
        ])
    ]
}

# 12 类疾病（全景片）
diseases_12_categories = {
    "categories": [
        {"id": 1, "name": "Impacted Tooth"},
        {"id": 2, "name": "Caries"},
        {"id": 3, "name": "Filling"},
        {"id": 4, "name": "Periapical Lesion"},
        {"id": 5, "name": "Deep Caries"},
        {"id": 6, "name": "Residual Root"},
        {"id": 7, "name": "Implant"},
        {"id": 8, "name": "Crown"},
        {"id": 9, "name": "Residual Crown"},
        {"id": 10, "name": "Pontic"},
        {"id": 11, "name": "Prosthesis"},
        {"id": 12, "name": "Abutment"}
    ]
}

# 11 类疾病
diseases_11_categories = {
    "categories": [
        {"id": 1, "name": "Impacted"},
        {"id": 2, "name": "Caries"},
        {"id": 3, "name": "Filling"},
        {"id": 4, "name": "Periapical Lesion"},
        {"id": 5, "name": "Deep Caries"},
        {"id": 6, "name": "Residual Root"},
        {"id": 7, "name": "Implant"},
        {"id": 8, "name": "Crown"},
        {"id": 9, "name": "Residual Crown"},
        {"id": 10, "name": "Pontic"},
        {"id": 11, "name": "Prosthesis"}
    ]
}

# 4 类疾病
diseases_4_categories = {
    "categories": [
        {"id": 1, "name": "Impacted"},
        {"id": 2, "name": "Caries"},
        {"id": 3, "name": "Filling"},
        {"id": 4, "name": "Periapical Lesion"}
    ]
}

# 根尖片 6 类疾病
periapical_6_diseases = {
    "categories": [
        {"id": 1, "name": "Caries"},
        {"id": 2, "name": "Deep Caries"},
        {"id": 3, "name": "Filling"},
        {"id": 4, "name": "Periapical Lesion"},
        {"id": 5, "name": "Residual Root"},
        {"id": 6, "name": "Crown"}
    ]
}

# 根尖片 3 类病变
periapical_3_lesions = {
    "categories": [
        {"id": 1, "name": "Apical Periodontitis"},
        {"id": 2, "name": "Cyst"},
        {"id": 3, "name": "Granuloma"}
    ]
}

# 4 象限
quadrants_4_categories = {
    "categories": [
        {"id": 1, "name": "Upper Right"},
        {"id": 2, "name": "Upper Left"},
        {"id": 3, "name": "Lower Left"},
        {"id": 4, "name": "Lower Right"}
    ]
}

# 下颌管和上颌窦
mandibular_maxillary_categories = {
    "categories": [
        {"id": 1, "name": "Mandibular Canal"},
        {"id": 2, "name": "Maxillary Sinus"}
    ]
}

# 骨质流失
bone_loss_categories = {
    "categories": [
        {"id": 1, "name": "Bone Loss"}
    ]
}

# 龋齿和充填
caries_filling_categories = {
    "categories": [
        {"id": 1, "name": "Caries"},
        {"id": 2, "name": "Filling"}
    ]
}

# 保存所有类别文件
categories_to_save = {
    "32ToothID_category.json": tooth_32_categories,
    "12diseases_category.json": diseases_12_categories,
    "11diseases_category.json": diseases_11_categories,
    "4diseases_category.json": diseases_4_categories,
    "periapical_6diseases_category.json": periapical_6_diseases,
    "periapical_3lesions_category.json": periapical_3_lesions,
    "4quadrants_category.json": quadrants_4_categories,
    "mandibular_maxillary_category.json": mandibular_maxillary_categories,
    "bone_loss_category.json": bone_loss_categories,
    "caries_filling_category.json": caries_filling_categories,
}

def create_all_category_files():
    """创建所有类别文件"""
    print("🦷 创建类别映射文件...")
    
    for filename, categories in categories_to_save.items():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(categories, f, indent=2, ensure_ascii=False)
        print(f"✅ 创建: {filename}")
        print(f"   - 类别数: {len(categories['categories'])}")
    
    print(f"\n✅ 成功创建 {len(categories_to_save)} 个类别文件！")

if __name__ == "__main__":
    create_all_category_files()
