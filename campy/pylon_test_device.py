from pypylon import pylon
# Start Pylon system
pylon.pylon_tlfactory = pylon.TlFactory.GetInstance()
# Get all available cameras
devices = pylon.TlFactory.GetInstance().EnumerateDevices()
print(devices)
# Print information for each device
for device in devices:
 print("Device Name:", device.GetFriendlyName())
 print("Device IP:", device.GetIpAddress())
 print("Device Model:", device.GetModelName())
 print("Device Serial Number:", device.GetSerialNumber())
 print("Device Vendor Name:", device.GetVendorName())
 print()
# Create a camera object and open it
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
# Demonstrate some feature access (changing camera width)
new_width = camera.Width.GetValue() - camera.Width.GetInc()
if new_width >= camera.Width.GetMin():
 camera.Width.SetValue(new_width)
# Set the number of images to grab
numberOfImagesToGrab = 100
camera.StartGrabbingMax(numberOfImagesToGrab)
# Start a loop for grabbing images
while camera.IsGrabbing():
 grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
if grabResult.GrabSucceeded():
 # Print size of the grabbed image
 print("SizeX: ", grabResult.Width)
 print("SizeY: ", grabResult.Height)
 
 # Access pixel values of the grabbed image
 img = grabResult.Array
 print("Gray value of first pixel: ", img[0, 0])
grabResult.Release()
# Close the camera
camera.Close()