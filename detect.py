import cv2 as cv
import os
import json
from collections import Counter 

tr = str.maketrans("üöçşğıÜÖÇŞĞİ", "uocsgiUOCSGI") 

def masks(hsv):
    blue = cv.inRange(hsv, (110, 50, 50), (130, 255, 255))
    green = cv.inRange(hsv, (35, 52, 72), (85, 255, 255))  
    red = cv.bitwise_or(cv.inRange(hsv, (0, 50, 20), (10, 255, 255)),
                        cv.inRange(hsv, (170, 50, 20), (179, 255, 255)))
    return {"kırmızı": red, "yeşil": green, "mavi": blue}

def main():
    os.makedirs('outputs', exist_ok=True)
    sonuc = []

    for name in sorted(os.listdir('data')):                     
        if not name.lower().endswith(('.png', '.jpg', '.jpeg')):  
            continue                                            

        img = cv.imread(os.path.join('data', name))             
        if img is None:                                         
            print("okunamadı:", name)
            continue 

        h, w = img.shape[:2]
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        data = {"name": name, "resolution": {"width": w, "height": h}, "features": []}

        for color, m in masks(hsv).items():
            contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv.contourArea(contour) < 300: continue

                #kenar sayısı ve şekil tipi
                epsilon = 0.02 * cv.arcLength(contour, True) #epsilonun 0.02 olması doğru bi karar mı?
                approx = cv.approxPolyDP(contour, epsilon, True)
                vertices = len(approx)
                if vertices == 3:
                    shape = "Üçgen"
                elif vertices == 4:
                    shape = "Dörtgen"
                elif vertices == 5:
                    shape = "Beşgen"
                elif vertices == 6:
                    shape = "Altıgen"
                else:
                    shape = "Tanımsız"

                #merkez koordinat momenti
                M = cv.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv.drawContours(img, [contour], -1, (0, 255, 0), 2)
                    cv.circle(img, (cx, cy), 7, (0, 0, 255), -1)
                    cv.putText(img, f"center: {cx}, {cy}", (cx - 20, cy - 20),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    print(f"x: {cx} y: {cy}")

                    data["features"].append({  
                        "shape": shape,
                        "color": color,
                        "vertices": vertices,
                        "cx": cx,
                        "cy": cy
                    })

                #cv.drawContours(img, [approx], 0, (0, 255, 0), 2) hem yanlış hem gereksiz geldi approx/contour ayrışımına bak

                x0, y0 = int(approx[0][0][0]), int(approx[0][0][1])
                cv.putText(img, shape.translate(tr), (x0, y0 + 5),  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv.putText(img, color.translate(tr), (x0, y0 + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        #bu kısıma çalış
        gecerli = [f for f in data["features"] if f["shape"] != "Tanımsız"]
        data["total_shapes"] = len(gecerli)
        data["unclassified"] = len(data["features"]) - len(gecerli)
        data["distribution"] = dict(Counter(f"{f['color']}_{f['shape']}" for f in gecerli))

        sonuc.append(data)

        out_name = os.path.splitext(name)[0] + '_result.png' 
        cv.imwrite(os.path.join('outputs', out_name), img)

    with open("sonuclar.json", "w", encoding="utf-8") as f: 
        json.dump({"coordinate_system": "Başlangıç sol üst köşe; x sağa, y aşağı; birim piksel",
            "images": sonuc}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()