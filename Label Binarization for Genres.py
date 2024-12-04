# Convert genres to binary labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

# Show genre labels
print(mlb.classes_)
