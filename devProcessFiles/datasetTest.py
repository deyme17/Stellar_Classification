from app.utils.dataset import Dataset

ds = Dataset('data\\train.csv')

print('1. len test')
print(f'{ds.len()}\n')

print('2. columns test')
print(f'{ds.columns()}\n')

print('3. getitem test')
print(f'{ds.getitem(2)}\n')

print('4. getitems test')
print(f'{ds.get_items(5)}')