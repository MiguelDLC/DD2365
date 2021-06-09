for(int q=0; q<n; q++){
    for (int k=0; k<q; k++){
        double x_diff = pos[q][X] - pos[k][X]; 
        double y_diff = pos[q][Y] - pos[k][Y]; 
        double dist = sqrt(x_diff*x_diff + y_diff*y_diff); 
        double dist_cubed = dist*dist*dist; 
        forces[q][X] -= G*mass[q]*mass[k]/dist_cubed * x_diff; 
        forces[q][Y] -= G*mass[q]*mass[k]/dist_cubed * y_diff; 
    }
    for (int k=q+1; k<n; k++){
        double x_diff = pos[q][X] - pos[k][X]; 
        double y_diff = pos[q][Y] - pos[k][Y]; 
        double dist = sqrt(x_diff*x_diff + y_diff*y_diff); 
        double dist_cubed = dist*dist*dist; 
        forces[q][X] -= G*mass[q]*mass[k]/dist_cubed * x_diff; 
        forces[q][Y] -= G*mass[q]*mass[k]/dist_cubed * y_diff; 
    }
}