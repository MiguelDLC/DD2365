for(int q=0; q<n; q++){
    for (int k=q+1; k<n; k++){
        double x_diff = pos[q][X] - pos[k][X]; 
        double y_diff = pos[q][Y] - pos[k][Y]; 
        double dist = sqrt(x_diff*x_diff + y_diff*y_diff); 
        double dist_cubed = dist*dist*dist; 
        force_qk[X] = G*mass[q]*mass[k]/dist_cubed * x_diff; 
        force_qk[Y] = G*mass[q]*mass[k]/dist_cubed * y_diff;
        forces[q][X] += force_qk[X];
        forces[q][Y] += force_qk[Y];
        forces[k][X] -= force_qk[X];
        forces[k][Y] -= force_qk[Y];
    }
}