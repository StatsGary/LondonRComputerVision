# Facial detection code written for LondonR - Gary Hutson
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN


def get_the_mush(filename, result_list):

	data = plt.imread(filename)
	for i in range(len(result_list)):
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		plt.subplot(1, len(result_list), i+1)
		plt.axis('off')
		plt.imshow(data[y1:y2, x1:x2])

	plt.show()

filename = 'Mango.jpg'
pixels = plt.imread(filename)
detector = MTCNN()
faces = detector.detect_faces(pixels)
get_the_mush(filename, faces)