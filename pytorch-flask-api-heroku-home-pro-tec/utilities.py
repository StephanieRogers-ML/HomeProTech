def upload_file():
  if request.method == 'POST':
      if 'file' not in request.files:
          return redirect(request.url)
      file = request.files.get('file')
      if not file:
          return
      img_bytes = file.read()
      class_id, class_name = get_prediction(image_bytes=img_bytes)
      class_name = format_class_name(class_name)
      return render_template('result.html', class_id=class_id,
                             class_name=class_name)
  return render_template('index.html')


def get_model():
      model = models.densenet121(pretrained=True)
      model.eval()
      return model

def transform_image(image_bytes):
      my_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])
      image = Image.open(io.BytesIO(image_bytes))
      return my_transforms(image).unsqueeze(0)


def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name



def get_prediction(image_bytes):
  try:
      tensor = transform_image(image_bytes=image_bytes)
      outputs = model.forward(tensor)
  except Exception:
      return 0, 'error'
  _, y_hat = outputs.max(1)
  predicted_idx = str(y_hat.item())
  return imagenet_class_index[predicted_idx]
