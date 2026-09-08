
def merge(arr, low, mid, high):
    left = arr[low:mid + 1]
    right = arr[mid + 1:high + 1]

    i = 0
    j = 0
    k = low

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    #add any remaining elements from the subarrys
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1

def sort(arr, low, high):
    if low < high:
        mid = (low + high) // 2
        sort(arr, low, mid)
        sort(arr, mid + 1, high)
        merge(arr, low, mid, high)

if __name__ == "__main__":
    arr = [38, 27, 43, 3, 9, 82, 10]
    sort(arr, 0, len(arr) - 1)
    print("Sorted array is:", arr)
