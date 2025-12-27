"""确认机制自定义异常"""


class ConfirmationError(Exception):
    """确认机制基础异常"""
    pass


class ConfirmationNotFoundError(ConfirmationError):
    """确认不存在"""
    pass


class ConfirmationExpiredError(ConfirmationError):
    """确认已过期"""
    pass


class ConfirmationAlreadyResolvedError(ConfirmationError):
    """确认已被处理"""
    pass


class ExecutorNotFoundError(ConfirmationError):
    """执行器未注册"""
    pass
