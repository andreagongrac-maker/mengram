import * as vscode from 'vscode';
import * as path from 'path';
import { MengramViewProvider } from './MengramViewProvider';

export function activate(context: vscode.ExtensionContext) {
    const provider = new MengramViewProvider(context.extensionUri, context.secrets);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            MengramViewProvider.viewType,
            provider,
        ),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('mengram.setApiKey', async () => {
            const key = await vscode.window.showInputBox({
                prompt: 'Enter your Mengram API key (om-...)',
                password: true,
                placeHolder: 'om-...',
            });
            if (key !== undefined) {
                await context.secrets.store('mengram.apiKey', key);
                vscode.window.showInformationMessage(
                    key ? 'Mengram API key saved.' : 'Mengram API key cleared.',
                );
            }
        }),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('mengram.searchMemories', () => {
            provider.focusSearch();
        }),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('mengram.saveSelection', () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor.');
                return;
            }
            const text = editor.document.getText(editor.selection);
            if (!text) {
                vscode.window.showWarningMessage('No text selected.');
                return;
            }
            provider.sendSelectedText(
                text,
                editor.document.fileName,
                editor.document.languageId,
            );
        }),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('mengram.saveFile', () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor.');
                return;
            }
            const content = editor.document.getText();
            const name = path.basename(editor.document.fileName);
            provider.saveText(`File: ${name}\n\n${content}`);
        }),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('mengram.showStats', () => {
            provider.requestStats();
        }),
    );

    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100,
    );
    statusBarItem.text = '$(database) Mengram';
    statusBarItem.tooltip = 'Open Mengram panel';
    statusBarItem.command = 'workbench.view.extension.mengram-sidebar';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);
}

export function deactivate() {}
