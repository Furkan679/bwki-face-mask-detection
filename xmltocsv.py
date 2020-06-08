import xml.etree.ElementTree as ET
import csv

tree = ET.parse("datasets/labels/000_1ov3n5_0.xml")
root = tree.getroot()

# open a file for writing

mask_data = open('C:/Users/Furkan1/Desktop/bwki/test.csv', 'w')

# create the csv writer object

csvwriter = csv.writer(mask_data)

mask_head = []

count = 0

print(0)

for j in root.findall('object'):
	print(2)

	for i in j:
		for k in i:
			print(k.attrib)

	name = j.find('name')
	print(name)
	pose = j.attrib
	box = j.attrib

	print("Name: {}, Pose: {}, Bandbox: {}".format(name, pose, box))

	mask_head.append(name)
	mask_head.append(pose)
	mask_head.append(box)

	print(mask_head)

	csvwriter.writerow(mask_head)

"""
	for k in root.findall('bndbox'):
		print(2)
		xmin = k.find('xmin').tag
		ymin = k.find('ymin').tag
		xmax = k.find('xmax').tag
		ymax = k.find('ymax').tag
	
		mask_head.append(xmin)
		mask_head.append(ymin)
		mask_head.append(xmax)
		mask_head.append(ymax)
"""

	

mask_data.close()