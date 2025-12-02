function [pairwise_distances, output_cam_0_pairwise, output_cam_1_pairwise, ...
    output_cam_02_pairwise, output_cam_13_pairwise, ...
    output_cam_0_undistorted_pairwise, output_cam_1_undistorted_pairwise,...
    output_cam_02_undistorted_pairwise, output_cam_13_undistorted_pairwise, ...
    acceptable_reprojection_errors_pairwise, ...
    worldPoints_pairwise] = get_pairwise_distances_between_dots(worldPoints, ...
    imagePoints_a, imagePoints_a_undistorted, ...
    imagePoints_b, imagePoints_b_undistorted, ...
    imagePoints_av, imagePoints_av_undistorted, ...
    imagePoints_bv, imagePoints_bv_undistorted, ...
    acceptable_reprojection_errors, ...
    grid_size, grid_spacing, num_images_used)

if num_images_used == -1
    num_images_used = size(worldPoints,1);
end

num_grid_points = num_images_used * (grid_size(1) * grid_size(2)) * (grid_size(1) * grid_size(2) - 1) / 2;
pairwise_distances = zeros(num_grid_points,1);
output_cam_0_pairwise = zeros(num_grid_points,4);
output_cam_1_pairwise = zeros(num_grid_points,4);
output_cam_02_pairwise = zeros(num_grid_points,4);
output_cam_13_pairwise = zeros(num_grid_points,4);
output_cam_0_undistorted_pairwise = zeros(num_grid_points,4);
output_cam_1_undistorted_pairwise = zeros(num_grid_points,4);
output_cam_02_undistorted_pairwise = zeros(num_grid_points,4);
output_cam_13_undistorted_pairwise = zeros(num_grid_points,4);
acceptable_reprojection_errors_pairwise = zeros(num_grid_points,2); % Reprojection errors of the two points in the pair
worldPoints_pairwise = zeros(num_grid_points,6);
[XX, YY] = meshgrid(grid_spacing*(1:grid_size(1)), grid_spacing*(1:grid_size(2)));
XX = XX(:);
YY = YY(:);
idx = 1;

for im_id = 1:num_images_used
    display(['Analyzing image', num2str(im_id), ' of ', num2str(size(worldPoints, 1))]);
    worldPoints_grid = reshape(worldPoints{im_id}, [grid_size, 3]);
    for i = 1:size(worldPoints{im_id}, 1)-1
        for j = i+1:size(worldPoints{im_id}, 1)
            pairwise_distances(idx) = norm([XX(i) - XX(j), YY(i) - YY(j)]);
            output_cam_0_pairwise(idx, :) = [imagePoints_a{im_id}(i,:), imagePoints_a{im_id}(j,:)];
            output_cam_1_pairwise(idx, :) = [imagePoints_b{im_id}(i,:), imagePoints_b{im_id}(j,:)];
            output_cam_02_pairwise(idx, :) = [imagePoints_av{im_id}(i,:), imagePoints_av{im_id}(j,:)];
            output_cam_13_pairwise(idx, :) = [imagePoints_bv{im_id}(i,:), imagePoints_bv{im_id}(j,:)];
            output_cam_0_undistorted_pairwise(idx, :) = [imagePoints_a_undistorted{im_id}(i,:), imagePoints_a_undistorted{im_id}(j,:)];
            output_cam_1_undistorted_pairwise(idx, :) = [imagePoints_b_undistorted{im_id}(i,:), imagePoints_b_undistorted{im_id}(j,:)];
            output_cam_02_undistorted_pairwise(idx, :) = [imagePoints_av_undistorted{im_id}(i,:), imagePoints_av_undistorted{im_id}(j,:)];
            output_cam_13_undistorted_pairwise(idx, :) = [imagePoints_bv_undistorted{im_id}(i,:), imagePoints_bv_undistorted{im_id}(j,:)];
            acceptable_reprojection_errors_pairwise(idx, :) = [acceptable_reprojection_errors{im_id}(i), acceptable_reprojection_errors{im_id}(j)];
            worldPoints_pairwise(idx, :) = [worldPoints{im_id}(i,:), worldPoints{im_id}(j,:)];
            idx = idx + 1;
        end
    end
end
display('done')