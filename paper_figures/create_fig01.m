% =========================================================================
% 1D Baseline Model Diagram of Peristaltic Flow with Tumor Constriction
% =========================================================================
clear; clc; close all;

% 1. Figure Setup
fig = figure('Name', '1D Peristaltic Model', 'Color', 'w', 'Position', [100, 100, 1400, 500]);
hold on; axis equal; axis off;
xlim([-4, 23]);
ylim([-5.5, 4.5]);

% 2. Geometry Parameters
L = 15;                     % Physical domain length (label is x=1)
xt = 7.5;                   % Center of the tumor region
sigma = 2.5;                % Spread of the tumor region
R0 = 1.2;                   % Base radius
wave_amp = 0.15;            % Peristaltic wave amplitude
wave_length = 4.5;          % Peristaltic wavelength
tumor_depth = 0.55;         % Constriction depth (\delta)

% Calculate tube boundaries
x = linspace(0, L, 1000);
h = R0 + wave_amp * sin(2*pi*x/wave_length) - tumor_depth * exp(-((x-xt).^2)/(0.8*sigma^2));

% 3. Plot Tube Interior (Light Red/Pink Gradient Simulation)
fill([x, fliplr(x)], [h, fliplr(-h)], [1, 0.85, 0.85], 'EdgeColor', 'none');

% 4. Plot Tumor Cells
cell_x = linspace(xt - sigma + 0.3, xt + sigma - 0.3, 14);
for i = 1:length(cell_x)
    hx_val = interp1(x, h, cell_x(i));
    % Top cells
    draw_tumor_cell(cell_x(i), hx_val + 0.35 + rand*0.1, 0.35);
    if rand > 0.3
        draw_tumor_cell(cell_x(i) + 0.2, hx_val + 0.7 + rand*0.1, 0.35);
    end
    % Bottom cells
    draw_tumor_cell(cell_x(i), -hx_val - 0.35 - rand*0.1, 0.35);
    if rand > 0.3
        draw_tumor_cell(cell_x(i) - 0.2, -hx_val - 0.7 - rand*0.1, 0.35);
    end
end

% 5. Plot Tube Walls
plot(x, h, 'Color', [0.8 0.1 0.1], 'LineWidth', 2.5);
plot(x, -h, 'Color', [0.8 0.1 0.1], 'LineWidth', 2.5);

% 6. Plot Nanoparticles (with updated exclusion zones)
N_nano = 160; 
rng(42); 
xn_all = rand(1, N_nano) * L;
yn_all = (rand(1, N_nano)*2 - 1) .* interp1(x, h, xn_all) * 0.75; 

% Exclusion zones: remove nanoparticles that overlap critical labels
flow_zone = (xn_all > 3 & xn_all < 6.5 & yn_all > -0.9 & yn_all < 0);
h_zone = (xn_all > 7.3 & xn_all < 8.8 & yn_all > -0.5 & yn_all < 0.5);
delta_zone = (xn_all > xt-2.0 & xn_all < xt-0.2 & yn_all > 0.5 & yn_all < 2.5); % Cleared for delta
keep_idx = ~(flow_zone | h_zone | delta_zone);

xn = xn_all(keep_idx);
yn = yn_all(keep_idx);

scatter(xn, yn, 20, 'b', 'filled', 'MarkerEdgeColor', [0 0 0.5]);

% 7. Centerline
plot([-1, L+1], [0, 0], 'k-.', 'LineWidth', 1.2);

% =========================================================================
% 8. ANNOTATIONS & LABELS
% =========================================================================

% --- Bottom Axis ---
y_ax = -4.0;
plot([0, L+1.5], [y_ax, y_ax], 'k-', 'LineWidth', 1); 
draw_arrow(L+1.5, y_ax, L+2.5, y_ax, 'k', 1);         
text(L+2.8, y_ax, '$x$', 'Interpreter', 'latex', 'FontSize', 14);

% Ticks and vertical droplines
ticks = [0, xt-sigma, xt, xt+sigma, L];
labels = {'$0$', '$x_t - \sigma$', '$x_t$', '$x_t + \sigma$', '$1$'};
for i = 1:length(ticks)
    plot([ticks(i), ticks(i)], [y_ax-0.1, y_ax+0.1], 'k-', 'LineWidth', 1);
    text(ticks(i), y_ax-0.6, labels{i}, 'Interpreter', 'latex', 'HorizontalAlignment', 'center', 'FontSize', 12);
    
    hx_tick = interp1(x, h, ticks(i));
    plot([ticks(i), ticks(i)], [-hx_tick-0.1, y_ax+0.2], 'k--', 'Color', [0.6 0.6 0.6]);
end

% --- Inlet (Left Side) ---
text(0, 2.6, 'Inlet', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(0, 2.0, '$x = 0$', 'Interpreter', 'latex', 'HorizontalAlignment', 'center');
draw_arrow(-2.5, 0, -0.5, 0, 'b', 2.5);
text(-1.5, -0.8, {'$u(x,t), C(x,t),$'; '$T(x,t)$'}, 'Interpreter', 'latex', 'Color', 'b', 'HorizontalAlignment', 'center', 'FontSize', 11);

% --- Outlet (Right Side) ---
text(L, 2.6, 'Outlet', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(L, 2.0, '$x = 1$', 'Interpreter', 'latex', 'HorizontalAlignment', 'center');
draw_arrow(L+0.5, 0, L+2.5, 0, 'b', 2.5);

% --- Tumor Region Constriction ---
br_y = 3.5; 
plot([xt-sigma, xt+sigma], [br_y, br_y], 'r--', 'LineWidth', 1.2); 
plot([xt-sigma, xt-sigma], [br_y, br_y-0.4], 'r--', 'LineWidth', 1.2); 
plot([xt+sigma, xt+sigma], [br_y, br_y-0.4], 'r--', 'LineWidth', 1.2); 
text(xt, br_y+0.4, 'Tumor region (constriction)', 'Color', 'r', 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'FontSize', 11);

% --- Peristaltic Wall ---
pw_x = 11.5; pw_y = 3.2;
text(pw_x, pw_y, 'Peristaltic wall', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(pw_x, pw_y-0.5, '$Y = \pm h(x)$', 'Interpreter', 'latex', 'HorizontalAlignment', 'center', 'FontSize', 11);
draw_arrow(pw_x-0.2, pw_y-0.8, pw_x-0.8, interp1(x, h, pw_x-0.8) + 0.1, 'k', 1.2);

% --- Flow Direction ---
fd_x = 3.5; fd_y = -0.3; 
draw_arrow(fd_x, fd_y, fd_x+2.5, fd_y, 'b', 2);
text(fd_x+1.25, fd_y-0.4, 'Flow direction', 'Color', 'b', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% --- Nanoparticles Label ---
np_x = 3.0; np_y = 2.8;
text(np_x, np_y, 'Nanoparticles', 'Color', 'b', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
draw_arrow(np_x, np_y-0.3, np_x+0.5, 0.8, 'k', 1);

% --- 2h(x) Constriction Height ---
hx_center = interp1(x, h, xt);
draw_double_arrow(xt, -hx_center, hx_center, 'k', 1.5);
text(xt+0.5, 0, '$2h(x)$', 'Interpreter', 'latex', 'FontSize', 12, 'BackgroundColor', 'w', 'Margin', 1.5);

% --- Delta (Constriction Depth) ---
% Draw dashed line representing the unconstricted wall across the whole tumor region
x_ref = linspace(xt - sigma, xt + sigma, 200);
h_ref = R0 + wave_amp * sin(2*pi*x_ref/wave_length);
plot(x_ref, h_ref, 'k--', 'LineWidth', 1.5, 'Color', [0.3 0.3 0.3]);

% Position delta indicator further left so it has room to breathe
del_x = xt - 1.2; 
h_unconst_del = R0 + wave_amp * sin(2*pi*del_x/wave_length);
h_actual_del = interp1(x, h, del_x);

% Draw double arrow for delta (will dynamically scale using the updated helper function)
draw_double_arrow(del_x, h_actual_del, h_unconst_del, 'k', 1.5);

% Add label with a black bounding box so it pops as a distinct callout
text(del_x - 0.15, (h_unconst_del + h_actual_del)/2, '$\delta$', 'Interpreter', 'latex', ...
    'FontSize', 15, 'HorizontalAlignment', 'right', 'BackgroundColor', 'w', ...
    'EdgeColor', 'k', 'Margin', 2);

% --- Wave Frame Box ---
box_x = 18; box_y = -1.5; box_w = 4; box_h = 3;
rectangle('Position', [box_x, box_y, box_w, box_h], 'EdgeColor', 'k', 'LineStyle', '--', 'Curvature', 0.1, 'LineWidth', 1);
text(box_x+2, box_y+2.4, 'Wave frame', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(box_x+2, box_y+1.8, '$(x = X - ct)$', 'Interpreter', 'latex', 'HorizontalAlignment', 'center', 'FontSize', 11);

% Coordinate system inside Wave Frame
ax_x = box_x+0.8; ax_y = box_y+0.6;
draw_arrow(ax_x, ax_y, ax_x+1.2, ax_y, 'k', 1.5); 
draw_arrow(ax_x, ax_y, ax_x, ax_y+1.0, 'k', 1.5); 
text(ax_x+1.4, ax_y, '$x$', 'Interpreter', 'latex', 'FontSize', 11);
text(ax_x, ax_y+1.3, '$y$', 'Interpreter', 'latex', 'FontSize', 11);


% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function draw_arrow(x1, y1, x2, y2, color, lw)
    plot([x1, x2], [y1, y2], 'Color', color, 'LineWidth', lw);
    ang = atan2(y2-y1, x2-x1);
    arrow_len = 0.4;
    theta = pi/7;
    xa1 = x2 - arrow_len * cos(ang - theta);
    ya1 = y2 - arrow_len * sin(ang - theta);
    xa2 = x2 - arrow_len * cos(ang + theta);
    ya2 = y2 - arrow_len * sin(ang + theta);
    fill([x2, xa1, xa2], [y2, ya1, ya2], color, 'EdgeColor', color);
end

function draw_double_arrow(x, y1, y2, color, lw)
    % Ensure y1 is the bottom coordinate and y2 is the top coordinate
    if y1 > y2
        temp = y1; y1 = y2; y2 = temp;
    end
    
    plot([x, x], [y1, y2], 'Color', color, 'LineWidth', lw);
    
    % Dynamically scale the arrowhead length and width based on the gap distance
    % This prevents arrowheads from overlapping (creating the "star" artifact) in tight spaces.
    gap = y2 - y1;
    arrow_len = min(0.3, gap * 0.35); 
    w = min(0.15, gap * 0.2); 
    
    % Draw top arrowhead
    fill([x, x-w, x+w], [y2, y2-arrow_len, y2-arrow_len], color, 'EdgeColor', color);
    % Draw bottom arrowhead
    fill([x, x-w, x+w], [y1, y1+arrow_len, y1+arrow_len], color, 'EdgeColor', color);
end

function draw_tumor_cell(xc, yc, r)
    theta = linspace(0, 2*pi, 30);
    cx = xc + r*cos(theta);
    cy = yc + r*sin(theta);
    fill(cx, cy, [0.85, 0.65, 0.85], 'EdgeColor', [0.5, 0.2, 0.5], 'LineWidth', 0.8);
    scatter(xc, yc, 15, 'MarkerFaceColor', [0.4, 0.1, 0.4], 'MarkerEdgeColor', 'none');
    for a = 0:pi/4:(2*pi - pi/4)
        plot([xc+0.1*cos(a), xc+r*0.8*cos(a)], [yc+0.1*sin(a), yc+r*0.8*sin(a)], 'Color', [0.6, 0.3, 0.6], 'LineWidth', 0.5);
    end
end