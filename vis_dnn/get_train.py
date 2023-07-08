def exp_train(train_loader, model, loss_function, optimizer, scheduler):
    out_list = []
    lat_list = []
    lab_list = []
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        output, latent = model(inputs)
        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        # print(loss)
        total_loss += loss
        
        optimizer.step()
        scheduler.step()
        out_list.append(output.detach().numpy())
        lat_list.append(latent.detach().numpy())
        lab_list.append(labels.detach().numpy())
    # print(t, total_loss)
    return out_list, lat_list, lab_list, total_loss


def xor_train(train_loader, model, loss_function, optimizer, scheduler, t):
    model.train()
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        output, feature_list = model(inputs)
        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss
        
        optimizer.step()
        scheduler.step()
    print(t, total_loss)
    
    return total_loss

def get_threshold(train_loader, model):
    model.eval()
    red = 0
    yellow = 0
    num_red = 0
    num_yellow = 0
    for i, (inputs, labels) in enumerate(train_loader):
        output, _ = model(inputs)
        if labels.item()==0:
            red += output
            num_red += 1
        if labels.item()==1:
            yellow += output
            num_yellow +=1
    avg_red = red/num_red
    avg_yellow = yellow/num_yellow
    return ((avg_red+avg_yellow)/2)[0].item()
    
def xor_test(all_loader, train_loader, model):
    model.eval()
    train_out_list = []
    train_lat_list = []
    train_lab_list = []
    dot_out_list = []
    dot_lat_list = []
    for inputs in all_loader:
        output, feature_list = model(inputs)
        
        # for i in range(len(train_loader)):
        dot_out_list.append(output.detach().numpy())
        dot_lat_list.append(feature_list)
        # model.reset()
    
    for i, (inputs, labels) in enumerate(train_loader):
        output, feature_list = model(inputs)
       
        train_out_list.append(output.detach().numpy())
        train_lab_list.append(labels.detach().numpy())
        train_lat_list.append(feature_list)
        # model.reset()
    
    return dot_out_list, dot_lat_list, train_out_list, train_lat_list, train_lab_list