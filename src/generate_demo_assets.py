"""Generate crisp, high-contrast synthetic demonstration images for the 12 Demo Lab cases."""

import os
from PIL import Image, ImageDraw, ImageFilter, ImageFont

def generate_demo_assets(target_dir: str):
    os.makedirs(target_dir, exist_ok=True)

    # 1. Object Existence (Supported: Cat Present)
    img1 = Image.new("RGB", (300, 300), color=(245, 240, 230))
    d1 = ImageDraw.Draw(img1)
    # Draw simple stylized cat
    d1.ellipse([100, 120, 200, 220], fill=(210, 130, 40), outline=(100, 50, 10)) # body
    d1.ellipse([115, 70, 185, 140], fill=(210, 130, 40), outline=(100, 50, 10)) # head
    d1.polygon([(115, 80), (130, 40), (145, 80)], fill=(180, 100, 20)) # left ear
    d1.polygon([(155, 80), (170, 40), (185, 80)], fill=(180, 100, 20)) # right ear
    d1.ellipse([130, 95, 142, 107], fill=(30, 150, 50)) # eye
    d1.ellipse([158, 95, 170, 107], fill=(30, 150, 50)) # eye
    d1.text((100, 240), "Cat in room", fill=(50, 50, 50))
    img1.save(os.path.join(target_dir, "case1_cat_present.png"))

    # 2. Object Existence (Unsupported: No Phone)
    img2 = Image.new("RGB", (300, 300), color=(135, 206, 235)) # sky
    d2 = ImageDraw.Draw(img2)
    d2.rectangle([0, 180, 300, 300], fill=(34, 139, 34)) # grass
    # Draw a tree
    d2.rectangle([135, 120, 165, 220], fill=(139, 69, 19)) # trunk
    d2.ellipse([90, 40, 210, 150], fill=(46, 139, 87)) # foliage
    d2.text((80, 240), "Park landscape (no phone)", fill=(255, 255, 255))
    img2.save(os.path.join(target_dir, "case2_phone_absent.png"))

    # 3. Attribute (Correct: Red Car)
    img3 = Image.new("RGB", (300, 300), color=(230, 230, 230))
    d3 = ImageDraw.Draw(img3)
    d3.rectangle([0, 200, 300, 300], fill=(80, 80, 80)) # road
    # Car body (Bright Red)
    d3.rectangle([50, 140, 250, 190], fill=(220, 30, 30), outline=(150, 0, 0))
    d3.polygon([(90, 140), (120, 90), (190, 90), (220, 140)], fill=(200, 20, 20))
    # Windows (light cyan)
    d3.polygon([(100, 135), (125, 98), (185, 98), (205, 135)], fill=(180, 220, 240))
    # Wheels
    d3.ellipse([70, 175, 110, 215], fill=(20, 20, 20))
    d3.ellipse([190, 175, 230, 215], fill=(20, 20, 20))
    d3.text((90, 240), "Red Sports Car", fill=(255, 255, 255))
    img3.save(os.path.join(target_dir, "case3_red_car.png"))

    # 4. Attribute (Deliberately Incorrect: Blue Car claiming to be Yellow)
    img4 = Image.new("RGB", (300, 300), color=(235, 235, 235))
    d4 = ImageDraw.Draw(img4)
    d4.rectangle([0, 200, 300, 300], fill=(80, 80, 80))
    # Car body (Bright Deep Blue)
    d4.rectangle([50, 140, 250, 190], fill=(30, 80, 220), outline=(0, 40, 150))
    d4.polygon([(90, 140), (120, 90), (190, 90), (220, 140)], fill=(20, 70, 200))
    d4.polygon([(100, 135), (125, 98), (185, 98), (205, 135)], fill=(180, 220, 240))
    d4.ellipse([70, 175, 110, 215], fill=(20, 20, 20))
    d4.ellipse([190, 175, 230, 215], fill=(20, 20, 20))
    d4.text((85, 240), "Deep Blue Vehicle", fill=(255, 255, 255))
    img4.save(os.path.join(target_dir, "case4_blue_car.png"))

    # 5. Counting (Three Apples)
    img5 = Image.new("RGB", (300, 300), color=(240, 235, 225))
    d5 = ImageDraw.Draw(img5)
    # Bowl
    d5.chord([40, 130, 260, 250], start=0, end=180, fill=(180, 120, 70), outline=(120, 80, 40))
    # 3 Red Apples
    d5.ellipse([70, 110, 125, 165], fill=(220, 20, 20)) # Apple 1
    d5.ellipse([120, 95, 180, 155], fill=(210, 30, 30))  # Apple 2
    d5.ellipse([175, 110, 230, 165], fill=(225, 25, 25)) # Apple 3
    d5.text((90, 230), "3 Apples in Bowl", fill=(50, 50, 50))
    img5.save(os.path.join(target_dir, "case5_counting_apples.png"))

    # 6. Spatial Relation (Cup On Table)
    img6 = Image.new("RGB", (300, 300), color=(245, 245, 250))
    d6 = ImageDraw.Draw(img6)
    # Table top
    d6.rectangle([30, 170, 270, 200], fill=(139, 90, 43))
    d6.rectangle([50, 200, 75, 290], fill=(110, 70, 30))
    d6.rectangle([225, 200, 250, 290], fill=(110, 70, 30))
    # Coffee Cup on table
    d6.rectangle([120, 120, 170, 170], fill=(250, 250, 250), outline=(180, 180, 180))
    d6.arc([160, 130, 185, 160], start=270, end=90, fill=(180, 180, 180), width=3)
    d6.text((80, 220), "White Cup on Table", fill=(255, 255, 255))
    img6.save(os.path.join(target_dir, "case6_cup_on_table.png"))

    # 7. Action / Activity (Dog Running)
    img7 = Image.new("RGB", (300, 300), color=(140, 210, 245))
    d7 = ImageDraw.Draw(img7)
    d7.rectangle([0, 190, 300, 300], fill=(40, 160, 40)) # field
    # Stylized running dog
    d7.ellipse([100, 130, 190, 180], fill=(160, 95, 30)) # torso
    d7.ellipse([170, 110, 215, 150], fill=(160, 95, 30)) # head
    # running legs outstretched
    d7.line([(110, 165), (70, 200)], fill=(160, 95, 30), width=6)
    d7.line([(175, 165), (220, 195)], fill=(160, 95, 30), width=6)
    d7.text((80, 240), "Dog running across field", fill=(255, 255, 255))
    img7.save(os.path.join(target_dir, "case7_dog_running.png"))

    # 8. Scene Recognition (Airport Runway)
    img8 = Image.new("RGB", (300, 300), color=(120, 170, 230))
    d8 = ImageDraw.Draw(img8)
    # Runway asphalt
    d8.polygon([(0, 300), (300, 300), (190, 130), (110, 130)], fill=(60, 60, 65))
    # Runway centerlines
    d8.line([(150, 140), (150, 160)], fill=(255, 255, 255), width=3)
    d8.line([(150, 180), (150, 210)], fill=(255, 255, 255), width=5)
    d8.line([(150, 235), (150, 280)], fill=(255, 255, 255), width=8)
    # Control tower
    d8.rectangle([30, 70, 60, 130], fill=(200, 200, 210))
    d8.polygon([(20, 70), (70, 70), (45, 50)], fill=(180, 50, 50))
    d8.text((100, 80), "Airport Runway", fill=(255, 255, 255))
    img8.save(os.path.join(target_dir, "case8_airport_scene.png"))

    # 9. Text / OCR (Exit Sign)
    img9 = Image.new("RGB", (300, 300), color=(40, 40, 45))
    d9 = ImageDraw.Draw(img9)
    # Exit sign box
    d9.rectangle([50, 100, 250, 200], fill=(220, 20, 20), outline=(255, 255, 255), width=4)
    # Draw large letters EXIT
    d9.text((95, 125), "EXIT", fill=(255, 255, 255))
    d9.text((70, 230), "Emergency Exit Sign", fill=(180, 180, 180))
    img9.save(os.path.join(target_dir, "case9_exit_sign.png"))

    # 10. Compound Claim (Red car + dog)
    img10 = Image.new("RGB", (300, 300), color=(220, 225, 230))
    d10 = ImageDraw.Draw(img10)
    d10.rectangle([0, 180, 300, 300], fill=(70, 70, 70))
    # Red car on left
    d10.rectangle([20, 150, 160, 195], fill=(220, 30, 30))
    d10.polygon([(50, 150), (70, 115), (120, 115), (140, 150)], fill=(200, 20, 20))
    d10.ellipse([35, 185, 65, 215], fill=(20, 20, 20))
    d10.ellipse([115, 185, 145, 215], fill=(20, 20, 20))
    # Dog on right
    d10.ellipse([190, 160, 240, 195], fill=(160, 95, 30))
    d10.ellipse([230, 145, 255, 170], fill=(160, 95, 30))
    d10.text((60, 240), "Red car parked with dog", fill=(255, 255, 255))
    img10.save(os.path.join(target_dir, "case10_compound_car_dog.png"))

    # 11. Unsupported Speculation (Laptop with price/date question)
    img11 = Image.new("RGB", (300, 300), color=(240, 240, 245))
    d11 = ImageDraw.Draw(img11)
    d11.rectangle([20, 180, 280, 290], fill=(160, 130, 90)) # wooden desk
    # Laptop base
    d11.polygon([(60, 220), (240, 220), (260, 240), (40, 240)], fill=(180, 185, 190))
    # Laptop screen
    d11.rectangle([(70, 100), (230, 220)], fill=(20, 25, 30), outline=(160, 165, 170), width=3)
    d11.text((70, 260), "Laptop on office desk", fill=(40, 40, 40))
    img11.save(os.path.join(target_dir, "case11_laptop_speculation.png"))

    # 12. Inconclusive (Foggy / Distant / Occluded)
    img12 = Image.new("RGB", (300, 300), color=(180, 185, 190))
    d12 = ImageDraw.Draw(img12)
    # Very faint, blurred circular silhouette in heavy fog
    d12.ellipse([120, 120, 180, 180], fill=(155, 160, 165))
    # Apply Gaussian blur for fog simulation
    img12 = img12.filter(ImageFilter.GaussianBlur(radius=8))
    d12_post = ImageDraw.Draw(img12)
    d12_post.text((60, 250), "Heavy Fog / Distant Object", fill=(100, 105, 110))
    img12.save(os.path.join(target_dir, "case12_foggy_occluded.png"))

    print(f"Generated 12 demonstration images in: {target_dir}")

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\aryay\Downloads\hallucination-firewall-main\hallucination-firewall-main\assets\demo_images"
    generate_demo_assets(target)
