from RPi import GPIO
import sys
if len(sys.argv)>1:
	level=GPIO.LOW
else:
	level=GPIO.HIGH

GPIO.setmode(GPIO.BOARD)
output_pin=18
GPIO.setup(output_pin, GPIO.OUT)
GPIO.output(output_pin,level)
lstr='on' if level==GPIO.HIGH else 'off'

print('GPIO set up successfully and turned {}'.format(lstr))
