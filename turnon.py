from RPi import GPIO
GPIO.setmode(GPIO.BOARD)
output_pin=18
GPIO.setup(output_pin, GPIO.OUT)
GPIO.output(output_pin,GPIO.HIGH)
print('GPIO set up successfully and turned on')
