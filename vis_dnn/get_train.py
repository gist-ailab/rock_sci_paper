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
    out_list = []
    lat_list = []
    lab_list = []
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        output, feature_list = model(inputs)
        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        # print(loss)
        total_loss += loss
        
        optimizer.step()
        scheduler.step()
        # for i in range(len(train_loader)):
        out_list.append(output.detach().numpy())
        lab_list.append(labels.detach().numpy())
        lat_list.append(feature_list)
        model.reset()
    print(t, total_loss)
    return out_list, lat_list, lab_list, total_loss
