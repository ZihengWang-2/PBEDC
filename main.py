import os
import torch
from MLclf import MLclf, transforms
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
import time
from datetime import datetime
import random
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import clip
from utils import ProgressBar, fp_dec2bin, fp_bin2dec, setup_seed, find_samples_greedy, find_samples_max_coverage

N = 1000
SEED = 20
RANDOM = False

def fault_injection(model, fault_rate):
    """
    Inject faults into the model parameters by flipping random bits.
    """
    random.seed(datetime.now())
    print('\nFault Injection\n')
    name_list = []
    parameter_list = []
    model_dict = model.state_dict()

    for name, parameters in model_dict.items():
        name_list.append(name)
        parameter_list.append(parameters)

    format = 16

    layer_list = range(488)
    name_list_new = []
    parameter_list_new = []
    parameter_count = 0
    for i in layer_list:
        name_list_new.append(name_list[i])
        parameter_list_new.append(parameter_list[i])
        temp = parameter_list[i].clone().detach().reshape(-1)
        parameter_count += temp.size(0)
    error_bits = int(parameter_count * fault_rate)

    for i in range(error_bits):
        name_index = random.randint(0, len(name_list_new) - 1)
        parameter = parameter_list_new[name_index]
        size = list(parameter.size())
        parameter = parameter.view(-1)
        weight_index = random.randint(0, len(parameter) - 1)
        weight = parameter[weight_index].item()
        weight_bin = list(fp_dec2bin(weight, format))

        for index, item in enumerate(weight_bin):
            weight_bin[index] = int(item)
        bit_index = random.randint(0, format - 1)
        weight_bin[bit_index] = 1 - weight_bin[bit_index]

        sign = weight_bin[0]
        exp = weight_bin[1:6]
        mant = weight_bin[6:16]
        for index, item in enumerate(exp):
            exp[index] = str(item)
        for index, item in enumerate(mant):
            mant[index] = str(item)
        exp_d = int("".join(exp),2)-15
        mant_d = int("".join(mant),2) / 2**10
        weight_dec = (-1)**sign * 2**exp_d * (1+mant_d)

        parameter[weight_index] = torch.tensor(weight_dec,dtype=torch.float16)

        parameter = parameter.view(size)
        model_dict[name_list[name_index]] = parameter

    return model_dict




def inference(dataset, model,fault_rate):
    with open('output.txt', 'a') as f:
        f.write('\nmodel:' + str(model))
        f.write('\nfault rate:' + str(fault_rate))
    f = open('similarity_1000_cifar100.txt', 'rb')
    golden = pickle.load(f)
    f.close()

    setup_seed(SEED, RANDOM)
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model, device)

    # Download the dataset
    if dataset == 'CIFAR100':
        cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=False, train=False)
    elif dataset == 'CIFAR10':
        cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=False, train=False)
    elif dataset == 'mini-imagenet':
        MLclf.miniimagenet_download(Download=True)
        miniimagenet, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(ratio_train=0.6, ratio_val=0.2,
                                                                                   seed_value=20, shuffle=True,
                                                                                     save_clf_data=True)
    else:
        print("invalid dataset")
    #del validation_dataset, test_dataset

    model_dict = fault_injection(model, fault_rate)
    setup_seed(SEED, RANDOM)
    model.load_state_dict(model_dict)

    print('\nInference\n')
    top1_acc = 0
    top5_acc = 0
    similarity_list = []
    for i in range(N):
        # Prepare the inputs
        if dataset == 'CIFAR100':
            image, class_id = cifar100[i]
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
        elif dataset == 'CIFAR10':
            image, class_id = cifar10[i]
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
        elif dataset == 'mini-imagenet':
            image, class_id = miniimagenet[i]
            class_id = class_id.numpy()
            image = transforms.ToPILImage(mode='RGB')(image)
            image_input = preprocess(image).unsqueeze(0).to(device)
            labels_to_marks = MLclf.labels_to_marks['mini-imagenet']


        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarity = similarity[0]
        values, indices = similarity.topk(5, largest=True, sorted=True)

        top1_acc += torch.eq(indices[0],class_id).item()
        top5_acc += torch.eq(indices,class_id).any().item()
        similarity_list.append(similarity)

    #f = open('similarity_1000_cifar100.txt', 'wb')
    #pickle.dump(similarity_list, f)
    #f.close()

    prediction_change_list = []
    compare_similarity = 0
    changes_list = []
    threshold = 0
    for i in range(N):
        if torch.argmax(similarity_list[i]) != torch.argmax(golden[i]):
            prediction_change_list.append(1)
        else:
            prediction_change_list.append(0)
        similarity_changes = torch.sum(torch.abs(similarity_list[i] - golden[i])).item()
        changes_list.append(similarity_changes)
        if similarity_changes > threshold:
            compare_similarity += 1
    compare_prediction = sum(prediction_change_list) / N
    compare_similarity = compare_similarity / N



    top1_acc = top1_acc / N
    top5_acc = top5_acc / N
    message = '\nTop 1 acc:{:.2f}. Top 5 acc:{:.2f}'.format(100 * top1_acc, 100 * top5_acc)
    #print(f'Top 1 acc:{100 * top1_acc:.2f}%; Top 5 acc:{100 * top5_acc:.2f}%')
    print(message)
    with open('output.txt', 'a') as f:
        f.write(message)

    return compare_prediction, compare_similarity, changes_list, prediction_change_list


if __name__ == '__main__':
    with open('output.txt', 'w') as f:
        f.write(' ')

    fault_rate = 1e-6
    #dataset = ['CIFAR100', 'CIFAR10', 'mini-imagenet']
    dataset = 'CIFAR100'
    #model_list = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
    model = 'RN50'
    compare_prediction_list = []
    compare_similarity_list = []
    changes_list_avg = [0] * N
    trials = 1000
    data = pd.DataFrame()
    softmax_changes = pd.DataFrame()
    prediction_changes = pd.DataFrame()
    progress = ProgressBar(trials)
    for i in range(trials):
        start = time.process_time()

        compare_prediction, compare_similarity, changes_list, prediction_change_list = inference(dataset,model,fault_rate)
        compare_prediction_list.append(compare_prediction)
        compare_similarity_list.append(compare_similarity)
        changes_list_avg += changes_list
        data[str(i)] = changes_list
        for j in range(len(changes_list)):
            if changes_list[j] > 0:
                changes_list[j] = 1
        softmax_changes[str(i)] = changes_list
        if pd.isnull(softmax_changes[str(i)]).any():
            softmax_changes[str(i)] = softmax_changes[str(i-1)].values
        prediction_changes[str(i)] = prediction_change_list

        end = time.process_time()
        print('Running time: %s Seconds' % (end - start))
        progress.current += 1
        progress()
    progress.done()


    changes_list_avg = np.divide(np.array(changes_list_avg),N)
    df = pd.DataFrame({
        'output': compare_similarity_list,
        'prediction': compare_prediction_list,
    }, index=range(trials))
    df.index.name = '#'
    writer = pd.ExcelWriter("output.xlsx")
    df.to_excel(writer,
                sheet_name='rate',
                float_format='%.4f',
                na_rep='nan')
    #plt.hist(changes_list_avg, bins=20)
    #plt.savefig('hist.jpg')

    prediction_changes.to_excel(writer,
                  sheet_name='prediction_changes',
                  float_format='%.4f',
                  na_rep='nan')

    softmax_changes.to_excel(writer,
                  sheet_name='softmax_changes',
                  float_format='%.4f',
                  na_rep='nan')
    writer.save()


    # find samples
    max_samples = 10
    file_name = 'output.xlsx'
    node_values = pd.read_excel(file_name, 2)
    array_outputs = np.delete(node_values.values, 0, axis=1)

    # Option 1: Use greedy search
    print("Using Greedy Search:")
    result_greedy, detection_greedy, count_greedy = find_samples_greedy(array_outputs, max_samples)
    print("Selected Samples:", result_greedy)
    print("Cumulative Coverage:", detection_greedy)
    print("Uncovered Errors:", count_greedy)

    # Save Greedy Search results
    greedy_results = pd.DataFrame({
        "Selected Samples": result_greedy,
        "Cumulative Coverage": detection_greedy
    })
    greedy_results.to_csv("greedy_results.csv", index=False)
    with open("greedy_summary.txt", "w") as f:
        f.write(f"Uncovered Errors: {count_greedy}\n")

    # Option 2: Use max overlapped coverage
    print("\nUsing Max Overlapped Coverage:")
    result_max, detection_max, count_max = find_samples_max_coverage(array_outputs, max_samples)
    print("Selected Samples:", result_max)
    print("Cumulative Coverage:", detection_max)
    print("Uncovered Errors:", count_max)

    # Save Max Overlapped Coverage results
    max_results = pd.DataFrame({
        "Selected Samples": result_max,
        "Cumulative Coverage": detection_max
    })
    max_results.to_csv("max_coverage_results.csv", index=False)
    with open("max_coverage_summary.txt", "w") as f:
        f.write(f"Uncovered Errors: {count_max}\n")