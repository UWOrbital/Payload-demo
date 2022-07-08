from PIL import Image

from decode import decode

args = decode.decoder_arguments()
with open(args.codes_filename, 'rb') as codes_file:
    decode.load_codes(codes_file)
with open(args.filename, 'rb') as data_file:
    data_stream = decode.Bitstream(data_file)
    img_arr = decode.decode(data_stream)
img = Image.fromarray(img_arr)
img.show()
