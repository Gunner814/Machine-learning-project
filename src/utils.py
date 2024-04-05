import matplotlib.pyplot as plt

def ShowLossAndAccuracy(epochs, model):
    fig , ax = plt.subplots(1,2)
    train_acc = model.history.history['accuracy']
    train_loss = model.history.history['loss']
    test_acc = model.history.history['val_accuracy']
    test_loss = model.history.history['val_loss']

    fig.set_size_inches(20,6)
    ax[0].plot(epochs , train_loss , label = 'Training Loss',marker='o', linewidth=2)
    ax[0].plot(epochs , test_loss , label = 'Testing Loss',marker='.', linewidth=2)
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs , train_acc , label = 'Training Accuracy',marker='o', linewidth=2)
    ax[1].plot(epochs , test_acc , label = 'Testing Accuracy',marker='.', linewidth=2)
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")

    plt.subplots_adjust(wspace=0.3)
