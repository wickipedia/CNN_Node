import numpy as np
import csv


def compute_kalman_gains(q=[5,0.2], r=[5, 0.2], p0=[5, 0.2], step_range=100):
	F=np.eye(2)
	H=np.eye(2)
	Q=np.array([[q[0],0],[0,q[1]]])
	R=np.array([[r[0],0],[0,r[1]]])
	P0=np.array([[p0[0],0],[0,p0[1]]])
	Pupd = None
	with open('kalman_gain.csv', 'w') as csvfile:
		fieldnames = ['Steps','k00','k01','k10','k11']
		kalman_writer = csv.writer(csvfile, delimiter=',')
		kalman_writer.writerow(fieldnames)
		for k in range(1,step_range):
			if Pupd is None:
				Pupd = P0
			Pest = np.matmul(np.matmul(F,Pupd),np.transpose(F)) + Q
			S = np.matmul(np.matmul(H,Pest), np.transpose(H)) + R
			K = np.matmul(np.matmul(Pest,np.transpose(H)), np.linalg.inv(S))
			csvrow = list()
			csvrow.append(k)
			for val in np.nditer(K):
				csvrow.append(val.tolist())
			kalman_writer.writerow(csvrow)
			Pupd = np.matmul((np.eye(2) - np.matmul(K,H)), Pest)




if __name__ == '__main__':
	compute_kalman_gains()


# with open('names.csv', 'w') as csvfile:
#     fieldnames = ['first_name', 'last_name']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
#     writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
#     writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})

# 	with open('eggs.csv', 'rb') as csvfile:
# ...     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
# ...     for row in spamreader:
# ...         print ', '.join(row)

# spamwriter = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
#     spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
