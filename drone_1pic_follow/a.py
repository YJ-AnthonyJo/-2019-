from multiprocessing import Process
import time

def func1():
	for i in range(10):
		print("a")
		time.sleep(0.1)

def func2():
	for i in range(10):
		print("b")
		time.sleep(0.1)

#def func3():


if __name__ == '__main__':
	# start로 각 프로세스를 시작합니다. func1이 끝나지 않아도 func2가 실행됩니다.
	p1 = Process(target=func1) #함수 1을 위한 프로세스
	p2 = Process(target=func2) #함수 1을 위한 프로세스
	#p3 = Process(target=func3) #함수 1을 위한 프로세스

	p1.start()
	p2.start()
	#p3.start()
	for i in range(10):
		print("c")
		time.sleep(0.1)

	p1.join()
	p2.join()
	#p3.join()
